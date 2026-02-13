import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def compute_likert_score(r: float, p_r: float, coef: float, p_coef: float) -> int:
    """
    Map statistical evidence to a 0-100 Likert score.

    0  -> strong "No" (evidence that lower ratio is not associated with higher performance)
    100 -> strong "Yes" (evidence that lower ratio is associated with higher performance)
    """
    score = 50.0

    # Contribution from simple correlation
    if p_r < 0.05:
        # Effect size scaled into [0, 30]
        effect = min(abs(r), 0.5) / 0.5 * 30.0
        if r < 0:
            score += effect
        elif r > 0:
            score -= effect

    # Contribution from regression coefficient (controlling for covariates)
    if p_coef < 0.05:
        # Stronger weight because this adjusts for confounders; up to +/- 30
        effect = min(abs(coef), 3.0) / 3.0 * 30.0
        if coef < 0:
            score += effect
        elif coef > 0:
            score -= effect

    # Clamp to [0, 100] and return as int
    score = max(0.0, min(100.0, score))
    return int(round(score))


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Student-teacher ratio: students per teacher (class size proxy)
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Keep relevant columns and drop missing values
    cols = [
        "stratio",
        "avg_score",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
    ]
    df_model = df[cols].dropna()

    n_obs = int(df_model.shape[0])

    # Simple Pearson correlation between ratio and performance
    r, p_r = stats.pearsonr(df_model["stratio"], df_model["avg_score"])

    # Linear regression controlling for observable covariates
    X = df_model[
        [
            "stratio",
            "calworks",
            "lunch",
            "english",
            "income",
            "expenditure",
            "computer",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = df_model["avg_score"]

    model = sm.OLS(y, X).fit()
    coef_stratio = float(model.params["stratio"])
    p_coef_stratio = float(model.pvalues["stratio"])

    # Likert score summarizing strength and direction of evidence
    response_score = compute_likert_score(r, p_r, coef_stratio, p_coef_stratio)

    # Build concise textual explanation
    direction = "negative" if r < 0 else "positive"
    association_word = "higher" if r < 0 else "lower"

    if (p_r < 0.05) or (p_coef_stratio < 0.05):
        evidence_phrase = "the direction and statistical significance"
    else:
        evidence_phrase = "their small magnitude and lack of statistical significance"

    explanation = (
        "Using data on {n} California school districts, I computed the student–teacher "
        "ratio as students per teacher and average academic performance as the mean of "
        "reading and math test scores. The simple Pearson correlation between the "
        "student–teacher ratio and average test scores is {r:.3f} (p = {p_r:.3g}), "
        "indicating a {direction} association where lower student–teacher ratios tend "
        "to be associated with {association_word} test scores. "
        "A linear regression of average scores on the student–teacher ratio, controlling "
        "for CalWorks participation, free-lunch eligibility, English-learner share, "
        "family income, per-pupil expenditure, and computers per student, yields a "
        "coefficient on the student–teacher ratio of {coef:.3f} (p = {p_coef:.3g}), "
        "which quantifies the association after adjusting for these observed differences. "
        "Based on {evidence_phrase} of these estimates, I interpret the evidence as "
        "{strength} that lower student–teacher "
        "ratios are associated with higher academic performance, summarized by the "
        "Likert-scale response value."
    ).format(
        n=n_obs,
        r=r,
        p_r=p_r,
        direction=direction,
        association_word=association_word,
        coef=coef_stratio,
        p_coef=p_coef_stratio,
        evidence_phrase=evidence_phrase,
        strength=(
            "strong"
            if response_score >= 75
            else "moderate"
            if response_score >= 60
            else "weak"
            if response_score >= 50
            else "little"
        ),
    )

    conclusion = {"response": response_score, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
