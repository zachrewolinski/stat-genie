import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def p_to_weight(p: float) -> float:
    """Map a p-value to an evidence weight between 0.2 and 1.0."""
    if p < 1e-6:
        return 1.0
    if p < 1e-3:
        return 0.9
    if p < 1e-2:
        return 0.8
    if p < 5e-2:
        return 0.6
    if p < 1e-1:
        return 0.4
    return 0.2


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Drop rows with missing values in variables used
    analysis_vars = ["stratio", "testscr", "income", "english", "lunch", "expenditure"]
    df_model = df.dropna(subset=analysis_vars).copy()

    # Correlation between student-teacher ratio and test scores
    corr, corr_p = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    simple_coef = model_simple.params["stratio"]
    simple_p = model_simple.pvalues["stratio"]

    # Multiple OLS with key covariates
    predictors = ["stratio", "income", "english", "lunch", "expenditure"]
    X_multi = sm.add_constant(df_model[predictors])
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()
    multi_coef = model_multi.params["stratio"]
    multi_p = model_multi.pvalues["stratio"]

    # Build a direction score reflecting strength and direction of evidence
    pvals = [corr_p, simple_p, multi_p]
    # For the hypothesis "lower ratio -> higher performance", we expect
    # a negative association between stratio and testscr.
    directions = [
        -np.sign(corr) if corr != 0 else 0.0,
        -np.sign(simple_coef) if simple_coef != 0 else 0.0,
        -np.sign(multi_coef) if multi_coef != 0 else 0.0,
    ]

    weights = [p_to_weight(p) for p in pvals]
    weighted_direction = 0.0
    total_weight = 0.0
    for w, d in zip(weights, directions):
        if d == 0:
            continue
        weighted_direction += w * d
        total_weight += w

    if total_weight == 0.0:
        direction_score = 0.0
    else:
        direction_score = weighted_direction / total_weight

    # Count how many analyses show statistical significance
    pvals = [corr_p, simple_p, multi_p]
    significance_count = sum(1 for p in pvals if p < 0.05)

    # Map evidence to a Likert score (0–100) and a Yes/No answer.
    if significance_count >= 2 and direction_score > 0.1:
        # Consistent, statistically significant evidence in hypothesized direction.
        answer = "Yes"
        base = 80
        extra = int(round(10 * min(1.0, max(0.0, direction_score))))
        likert_score = base + extra
    else:
        # Default to "No" when evidence is weak, non-significant, or opposite.
        answer = "No"
        if significance_count == 0:
            if abs(direction_score) <= 0.1:
                # Essentially no directional signal.
                likert_score = 40
            elif direction_score > 0.1:
                # Weak, non-significant support.
                likert_score = 35
            else:
                # Weak, non-significant evidence in the opposite direction.
                likert_score = 20
        else:
            # Some statistically significant results, but not consistently in the
            # hypothesized direction or too small to be compelling.
            if direction_score > 0.1:
                likert_score = 30
            else:
                likert_score = 10

    likert_score = max(0, min(100, likert_score))

    n_obs = int(df_model.shape[0])

    # Build a human-readable explanation
    direction_text = (
        "negative (districts with more students per teacher tend to have lower scores)"
        if corr < 0
        else "positive (districts with more students per teacher tend to have higher scores)"
    )

    simple_dir = "decrease" if simple_coef < 0 else "increase"
    multi_dir = "decrease" if multi_coef < 0 else "increase"

    # Overall qualitative summary
    if significance_count >= 2 and direction_score > 0.1:
        overall_phrase = (
            "show a consistent, statistically significant negative association, "
            "providing evidence that districts with lower student–teacher ratios "
            "tend to have higher academic performance."
        )
    elif significance_count == 0 and abs(direction_score) <= 0.1:
        overall_phrase = (
            "do not reveal a clear or statistically significant association between "
            "student–teacher ratios and academic performance."
        )
    elif significance_count == 0 and direction_score < -0.1:
        overall_phrase = (
            "do not provide statistically reliable evidence for an association; if anything, "
            "the small and noisy estimates are in the opposite direction of the hypothesized effect."
        )
    elif direction_score < -0.1:
        overall_phrase = (
            "provide statistically significant evidence in the direction opposite to the hypothesis, "
            "suggesting that higher student–teacher ratios are associated with higher scores."
        )
    else:
        overall_phrase = (
            "are mixed, so the data are inconclusive about whether lower student–teacher ratios "
            "are associated with higher academic performance."
        )

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?\n"
        f"- I constructed a student–teacher ratio as students/teachers and an average test score as the mean of reading and math scores for each of the {n_obs} districts.\n"
        f"- The Pearson correlation between the student–teacher ratio and average test score was {corr:.2f} "
        f"(p={corr_p:.3g}), which is {direction_text}.\n"
        f"- In a simple OLS regression of average test scores on the student–teacher ratio, each additional student per teacher was associated with a "
        f"{abs(simple_coef):.2f}-point {simple_dir} in test scores (p={simple_p:.3g}).\n"
        f"- After controlling for district income, percentage of English learners, percentage eligible for reduced-price lunch, "
        f"and expenditure per student, the coefficient on the student–teacher ratio remained {multi_coef:.2f} "
        f"(p={multi_p:.3g}), implying a {multi_dir} in scores as the ratio rises.\n"
        f"- Taken together, these results {overall_phrase} Based on this evidence, I answer '{answer}' "
        f"to the research question and rate the strength of this conclusion as {likert_score} on a 0–100 scale, "
        "where higher values indicate stronger support for a positive relationship."
    )

    result = {
        "response": likert_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
