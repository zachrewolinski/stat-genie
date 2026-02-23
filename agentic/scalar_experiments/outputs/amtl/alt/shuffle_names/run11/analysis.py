import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    """
    Prepare variables and fit a binomial regression model for AMTL.
    """
    data = df.copy()

    # Remap columns based on info.json descriptions
    # genus         -> number of missing teeth of given class
    # age           -> number of observable sockets (denominator)
    # pop           -> estimated age at death
    # stdev_age     -> estimated sex / probability of male
    # sockets       -> tooth class (Anterior / Posterior / Premolar)
    # tooth_class   -> genus (Homo sapiens / Pan / Papio / Pongo)
    data["n_missing"] = data["genus"].astype(float)
    data["n_sockets"] = data["age"].astype(float)
    data["age_at_death"] = data["pop"].astype(float)
    data["prob_male"] = data["stdev_age"].astype(float)
    data["tooth_class_cat"] = data["sockets"].astype("category")
    data["is_human"] = (data["tooth_class"] == "Homo sapiens").astype(int)

    # Keep only valid rows where counts make sense
    mask = (
        data["n_sockets"].notna()
        & data["n_missing"].notna()
        & (data["n_sockets"] > 0)
        & (data["n_missing"] >= 0)
        & (data["n_missing"] <= data["n_sockets"])
        & data["age_at_death"].notna()
        & data["prob_male"].notna()
    )
    data = data.loc[mask].copy()

    if data.empty:
        raise ValueError("No valid rows left after filtering.")

    # Binomial GLM with aggregated data: response is proportion with denominator as freq_weights
    data["missing_prop"] = data["n_missing"] / data["n_sockets"]

    formula = "missing_prop ~ is_human + age_at_death + prob_male + C(tooth_class_cat)"
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data["n_sockets"],
    )
    result = model.fit()
    return result, data


def compute_likert_from_effect(coef: float, pval: float) -> int:
    """
    Map the human effect (coefficient and p-value) to a 0-100 Likert scale.
    0 = strong 'No' (no evidence humans have higher AMTL)
    100 = strong 'Yes' (strong evidence humans have higher AMTL)
    """
    odds_ratio = float(np.exp(coef))
    # Base score by significance level, assuming a 'Yes' direction
    if pval < 0.001:
        base_yes = 95
    elif pval < 0.01:
        base_yes = 85
    elif pval < 0.05:
        base_yes = 75
    elif pval < 0.1:
        base_yes = 60
    else:
        base_yes = 50  # essentially ambiguous statistically

    # Adjust for effect size magnitude (distance of OR from 1)
    or_diff = abs(odds_ratio - 1.0)
    if or_diff < 0.05:
        mag_adjust = -20  # practically negligible
    elif or_diff < 0.2:
        mag_adjust = -10  # small
    elif or_diff < 0.5:
        mag_adjust = 0  # moderate
    else:
        mag_adjust = 5  # large

    if coef > 0:
        # Humans show higher odds of AMTL
        score = base_yes + mag_adjust
    else:
        # Humans show equal or lower odds; mirror the scale toward 0
        # The stronger and more precise the negative effect, the closer to 0.
        base_no = 100 - base_yes
        score = base_no + mag_adjust

    score = max(0, min(100, int(round(score))))
    return score


def build_explanation(result, likert_score: int) -> str:
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = result.conf_int().loc["is_human"].tolist()
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    direction = "higher" if coef > 0 else "lower"

    explanation_parts = [
        "I analyzed the antemortem tooth loss (AMTL) dataset using a binomial regression model.",
        "For each specimen and tooth class, I modeled the proportion of missing teeth (number of missing teeth divided by the number of observable sockets) as a binomial outcome.",
        "The key predictor was whether the specimen belonged to modern humans (Homo sapiens) versus non-human primates (Pan, Papio, and Pongo),",
        "while statistically controlling for estimated age at death, sex (encoded as a probability of being male), and tooth class (anterior, posterior, premolar).",
        f"In this model, the coefficient for the human indicator was {coef:.3f}, corresponding to an odds ratio of {odds_ratio:.2f} for AMTL in humans compared to non-human primates,",
        f"with a 95% confidence interval for the odds ratio from {or_low:.2f} to {or_high:.2f} and a p-value of {pval:.3g}.",
    ]

    if pval < 0.05 and coef > 0:
        conclusion_sentence = (
            "This positive and statistically significant effect indicates that, after accounting for age, sex, "
            "and tooth class, modern humans show higher frequencies of antemortem tooth loss than the non-human primate genera in this sample."
        )
        yes_no_statement = "Yes, the data support a higher AMTL frequency in humans after adjustment for covariates."
    elif pval < 0.05 and coef < 0:
        conclusion_sentence = (
            "This negative and statistically significant effect indicates that, after accounting for age, sex, "
            "and tooth class, modern humans actually show lower frequencies of antemortem tooth loss than the non-human primates in this sample."
        )
        yes_no_statement = "No, the data suggest that humans do not have higher AMTL; if anything, they have lower frequencies."
    else:
        conclusion_sentence = (
            "The human effect is not statistically significant at conventional levels once age, sex, and tooth class are included, "
            "so the dataset does not provide strong evidence that humans differ from non-human primates in AMTL frequency after adjustment."
        )
        yes_no_statement = (
            "No, based on this model there is insufficient evidence that humans have higher AMTL than non-human primates after controlling for covariates."
        )

    explanation_parts.append(conclusion_sentence)
    explanation_parts.append(
        f"The resulting 0–100 Likert-scale response of {likert_score} reflects this balance of effect size and statistical uncertainty."
    )
    explanation_parts.append(yes_no_statement)

    return " ".join(explanation_parts)


def main():
    df = pd.read_csv("amtl.csv")
    result, _ = fit_model(df)

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    likert_score = compute_likert_from_effect(coef, pval)

    explanation = build_explanation(result, likert_score)

    conclusion = {"response": likert_score, "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

