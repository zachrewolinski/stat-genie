import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only columns relevant to the stated research question.
    cols = [
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "genus",
    ]
    df = df[cols].copy()

    # Drop obviously invalid or unusable rows.
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    # Sockets must be positive for a meaningful rate.
    df = df[df["sockets"] > 0].copy()

    # For binomial modeling, counts should not exceed the number of trials.
    # We exclude rows that violate this constraint and record how many were removed.
    valid_mask = df["num_amtl"] <= df["sockets"]
    df = df[valid_mask].copy()

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure categorical treatment for tooth class.
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Proportion of antemortem tooth loss per observable socket.
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Grouped-binomial GLM: model the log-odds of AMTL per socket.
    # Response is proportion with the number of sockets used as weights.
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def compute_likert_from_result(result) -> tuple[int, str]:
    coef = result.params.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    if np.isnan(coef) or np.isnan(pval):
        response = 50
        explanation = (
            "The model could not estimate a stable effect for modern humans "
            "relative to non-human primates, so the evidence is inconclusive."
        )
        return response, explanation

    odds_ratio = float(np.exp(coef))
    conf_int = result.conf_int().loc["is_human"]
    or_low = float(np.exp(conf_int[0]))
    or_high = float(np.exp(conf_int[1]))

    # Map the evidence to a 0–100 Likert scale where higher values mean
    # stronger support for the claim that humans have *higher* AMTL than
    # non-human primates.
    if coef > 0:
        # Positive coefficient: possible evidence that humans have higher AMTL.
        if pval < 0.001:
            response = 95
        elif pval < 0.01:
            response = 85
        elif pval < 0.05:
            response = 75
        elif pval < 0.1:
            response = 60
        else:
            # Positive but not statistically significant: weak evidence for "Yes".
            response = 40
    elif coef < 0:
        # Negative coefficient: evidence against higher AMTL in humans.
        if pval < 0.001:
            response = 5
        elif pval < 0.01:
            response = 10
        elif pval < 0.05:
            response = 20
        elif pval < 0.1:
            response = 30
        else:
            # Negative but not statistically significant: we still answer "No",
            # but with modest strength because the estimate is very uncertain.
            response = 25
    else:
        # Exactly zero effect estimate (rare in practice): completely neutral.
        response = 50

    # Build human-readable explanation.
    significance = (
        "strongly statistically significant (p < 0.001)"
        if pval < 0.001
        else "statistically significant (p < 0.01)"
        if pval < 0.01
        else "statistically significant (p < 0.05)"
        if pval < 0.05
        else "marginally significant (p < 0.1)"
        if pval < 0.1
        else "not statistically significant (p ≥ 0.1)"
    )

    if coef > 0 and pval < 0.05:
        qualitative = (
            "provides statistically reliable evidence that modern humans have higher "
            "frequencies of antemortem tooth loss than non-human primates after "
            "accounting for age, sex, and tooth class."
        )
    elif coef > 0 and pval >= 0.05:
        qualitative = (
            "suggests slightly higher frequencies of antemortem tooth loss in modern "
            "humans, but this difference is not statistically reliable and could easily "
            "be due to sampling variation."
        )
    elif coef < 0 and pval < 0.05:
        qualitative = (
            "provides statistically reliable evidence that modern humans actually have "
            "lower frequencies of antemortem tooth loss than non-human primates after "
            "accounting for age, sex, and tooth class. For the specific question of "
            "whether humans have higher AMTL, this supports a strong 'No' answer."
        )
    else:
        # coef <= 0 and not statistically significant
        qualitative = (
            "does not provide evidence that modern humans have higher frequencies of "
            "antemortem tooth loss than non-human primates once age, sex, and tooth "
            "class are controlled. The estimated effect is small and statistically "
            "indistinguishable from no difference, so the appropriate answer to the "
            "research question is 'No'—we do not detect higher AMTL in humans."
        )

    explanation = (
        "I fit a grouped-binomial regression model of the proportion of antemortem "
        "tooth loss per observable socket, using a logit link and weighting each "
        "row by the number of sockets. The predictors were a binary indicator for "
        "modern humans versus non-human primates (Pan, Pongo, Papio), age at death, "
        "estimated probability of being male, and tooth class (anterior, posterior, "
        "premolar). Rows with invalid counts (num_amtl > sockets) or zero sockets "
        "were excluded so that the binomial model assumptions were satisfied. "
        f"The estimated coefficient for modern humans (vs. non-human primates) was {coef:.3f}, "
        f"corresponding to an odds ratio of {odds_ratio:.2f} "
        f"(95% CI [{or_low:.2f}, {or_high:.2f}]); this effect was {significance} "
        f"(p = {pval:.4f}). Overall, this model {qualitative} "
        "The Likert-scale response encodes how strongly the data support the specific "
        "claim that humans have higher AMTL than non-human primates, with values near "
        "0 representing a strong 'No' and values near 100 representing a strong 'Yes'."
    )

    return int(round(response)), explanation


def main():
    df = load_and_prepare_data("amtl.csv")
    result = fit_binomial_model(df)
    response, explanation = compute_likert_from_result(result)

    conclusion = {"response": response, "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
