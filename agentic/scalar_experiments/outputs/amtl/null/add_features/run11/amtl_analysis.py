import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing key fields, if any
    df = df.copy()
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Proportion of missing teeth
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Ensure non-zero sockets
    df = df[df["sockets"] > 0]

    # Categorical encodings
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Create indicator for humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression model for AMTL:
        logit(p(AMTL)) = beta0 + beta1 * is_human + beta2 * age + beta3 * prob_male
                         + tooth_class effects
    using sockets as binomial trials and num_amtl as successes.
    """
    # Use proportion as endog with var_weights = sockets, which is equivalent to binomial
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_human_effect(result) -> dict:
    # Extract coefficient and p-value for is_human
    params = result.params
    pvalues = result.pvalues

    coef = float(params.get("is_human", np.nan))
    pval = float(pvalues.get("is_human", np.nan))

    # Compute odds ratio
    odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

    return {
        "coef": coef,
        "pval": pval,
        "odds_ratio": odds_ratio,
    }


def map_to_likert(human_effect: dict) -> int:
    """
    Map statistical evidence about human vs non-human AMTL to a 0–100 Likert scale,
    where higher values indicate stronger evidence that humans have higher AMTL.
    """
    coef = human_effect["coef"]
    pval = human_effect["pval"]
    odds_ratio = human_effect["odds_ratio"]

    # Default neutral response if things go badly
    if not np.isfinite(coef) or not np.isfinite(pval) or not np.isfinite(odds_ratio):
        return 50

    # Strong evidence humans have higher AMTL
    if coef > 0 and pval < 0.001:
        if odds_ratio >= 2.0:
            return 95
        elif odds_ratio >= 1.5:
            return 85
        else:
            return 75

    # Moderate evidence humans have higher AMTL
    if coef > 0 and pval < 0.01:
        return 70

    # Weak but significant evidence humans have higher AMTL
    if coef > 0 and pval < 0.05:
        return 60

    # No convincing evidence for difference
    if pval >= 0.05:
        return 40 if coef > 0 else 30

    # Fallback
    return 50


def build_explanation(
    df: pd.DataFrame,
    result,
    human_effect: dict,
    likert_value: int,
) -> str:
    n_total = len(df)
    n_human = int((df["genus"] == "Homo sapiens").sum())
    n_nonhuman = n_total - n_human

    mean_prop_human = float(df.loc[df["is_human"] == 1, "prop_amtl"].mean())
    mean_prop_nonhuman = float(df.loc[df["is_human"] == 0, "prop_amtl"].mean())

    coef = human_effect["coef"]
    pval = human_effect["pval"]
    odds_ratio = human_effect["odds_ratio"]

    explanation = []
    explanation.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) compared to non-human primate genera "
        "(Pan, Pongo, Papio), after accounting for age, sex, and tooth class?"
    )
    explanation.append(
        f"The dataset contains {n_total} tooth-class observations from {n_human} human "
        f"and {n_nonhuman} non-human primate specimens. For each observation I used "
        "the number of missing teeth (`num_amtl`) out of the observable sockets "
        "(`sockets`) to define the AMTL proportion."
    )
    explanation.append(
        "I fit a binomial regression model with a logit link:\n"
        "  logit(p(AMTL)) = β0 + β1 * I(human) + β2 * age + β3 * prob_male "
        "+ tooth-class indicators.\n"
        "Here I(human) equals 1 for Homo sapiens and 0 for the non-human genera, "
        "so β1 captures the difference in AMTL between humans and non-human primates "
        "after adjusting for age, sex (via `prob_male`), and tooth class."
    )
    explanation.append(
        f"Descriptively, the mean AMTL proportion is "
        f"{mean_prop_human:.3f} in humans and {mean_prop_nonhuman:.3f} in "
        "non-human primates."
    )
    explanation.append(
        f"In the regression, the human indicator has coefficient β1 = {coef:.3f}, "
        f"corresponding to an odds ratio of approximately {odds_ratio:.2f}, "
        f"with p-value {pval:.3g}."
    )

    if pval < 0.001 and coef > 0:
        interpretation = (
            "This provides strong statistical evidence that, after controlling for "
            "age, sex, and tooth class, humans have higher odds of AMTL than "
            "non-human primates."
        )
    elif pval < 0.05 and coef > 0:
        interpretation = (
            "This provides statistically significant but more moderate evidence "
            "that humans have higher odds of AMTL than non-human primates after "
            "controlling for age, sex, and tooth class."
        )
    elif pval >= 0.05:
        interpretation = (
            "The human coefficient is not statistically significant at conventional "
            "levels, so the data do not provide strong evidence that humans differ "
            "from non-human primates in AMTL after controlling for age, sex, "
            "and tooth class."
        )
    else:
        interpretation = (
            "The direction of the human coefficient suggests higher AMTL in humans, "
            "but the evidence is weak."
        )

    explanation.append(interpretation)

    if likert_value >= 50:
        explanation.append(
            f"Based on this analysis, I answer 'Yes' to the research question: "
            f"humans do have higher AMTL frequencies than non-human primates "
            f"after accounting for age, sex, and tooth class. "
            f"The strength of this conclusion is summarized by a Likert-scale "
            f"response value of {likert_value} on a 0–100 scale, where higher "
            "values indicate stronger evidence for a 'Yes' answer."
        )
    else:
        explanation.append(
            "Based on this analysis, I answer 'No' to the research question: "
            "the data do not provide strong evidence that humans have higher "
            "AMTL frequencies than non-human primates after accounting for "
            "age, sex, and tooth class. "
            f"The strength of this conclusion is summarized by a Likert-scale "
            f"response value of {likert_value} on a 0–100 scale, where lower "
            "values indicate stronger evidence for a 'No' answer."
        )

    return "\n\n".join(explanation)


def write_conclusion(response_value: int, explanation: str, path: str = "conclusion.txt") -> None:
    obj = {"response": int(response_value), "explanation": explanation}
    Path(path).write_text(json.dumps(obj, ensure_ascii=False))


def main():
    df = load_data()
    df = prepare_data(df)
    result = fit_binomial_model(df)
    human_effect = summarize_human_effect(result)
    likert_value = map_to_likert(human_effect)
    explanation = build_explanation(df, result, human_effect, likert_value)
    write_conclusion(likert_value, explanation)


if __name__ == "__main__":
    main()

