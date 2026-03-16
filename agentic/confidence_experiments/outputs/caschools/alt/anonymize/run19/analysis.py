import json
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def compute_student_teacher_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute students-per-teacher ratio from enrollment and teacher counts."""
    ratio = df["feature6"] / df["feature7"]
    return ratio.replace([np.inf, -np.inf], np.nan)


def compute_academic_performance(df: pd.DataFrame) -> pd.Series:
    """Compute overall academic performance as the average of reading and math scores."""
    return df[["feature14", "feature15"]].mean(axis=1)


def summarize_association(
    ratio: pd.Series, performance: pd.Series
) -> Tuple[float, float, float, float, float]:
    """
    Return (corr, p_corr, coef, p_coef, r2) for the association between
    student–teacher ratio and academic performance.
    """
    valid = ratio.notna() & performance.notna()
    ratio_valid = ratio[valid]
    perf_valid = performance[valid]

    corr, p_corr = stats.pearsonr(ratio_valid, perf_valid)

    X = sm.add_constant(ratio_valid)
    model = sm.OLS(perf_valid, X).fit()
    # The first parameter is the intercept, the second is the slope on the ratio.
    coef = float(model.params.iloc[1])
    p_coef = float(model.pvalues.iloc[1])
    r2 = float(model.rsquared)
    return corr, p_corr, coef, p_coef, r2


def likert_score_from_association(corr: float, p_corr: float) -> int:
    """
    Map the strength and significance of the correlation to a 0–100 Likert score
    answering: "Is a lower student-teacher ratio associated with higher academic performance?"
    Negative correlations support a "Yes" answer (more students per teacher -> lower scores).
    """
    abs_r = abs(corr)

    # No statistically significant evidence
    if p_corr >= 0.05:
        if abs_r < 0.1:
            return 50  # essentially no detectable relationship
        if corr < 0:
            return 55  # weak, non-significant trend in expected direction
        return 45  # weak, non-significant trend in opposite direction

    # Statistically significant evidence; direction and magnitude matter.
    if corr < 0:  # expected direction: lower ratio -> higher performance
        if abs_r < 0.1:
            return 60  # very small but significant effect
        if abs_r < 0.3:
            return 75  # small-to-moderate, significant association
        if abs_r < 0.5:
            return 85  # moderate, clearly supported association
        return 95  # very strong, highly consistent association

    # Significant but in the opposite direction (higher ratios -> higher performance).
    if abs_r < 0.1:
        return 40  # minimal but statistically non-negligible effect
    if abs_r < 0.3:
        return 25  # small but consistent evidence against the hypothesis
    return 10  # substantial evidence in the opposite direction


def build_explanation(
    corr: float,
    p_corr: float,
    coef: float,
    p_coef: float,
    r2: float,
    likert: int,
) -> str:
    direction = (
        "lower student-teacher ratios (fewer students per teacher) are associated with higher test scores"
        if corr < 0
        else "higher student-teacher ratios are associated with higher test scores"
    )

    strength_desc: str
    abs_r = abs(corr)
    if abs_r < 0.1:
        strength_desc = "effectively no association in practical terms"
    elif abs_r < 0.3:
        strength_desc = "a small but non-negligible association"
    elif abs_r < 0.5:
        strength_desc = "a moderate association"
    else:
        strength_desc = "a strong association"

    if p_corr < 0.001:
        sig_desc = "highly statistically significant (p < 0.001)"
    elif p_corr < 0.01:
        sig_desc = "statistically significant at the 1% level (p < 0.01)"
    elif p_corr < 0.05:
        sig_desc = "statistically significant at the 5% level (p < 0.05)"
    else:
        sig_desc = "not statistically significant at conventional levels (p ≥ 0.05)"

    explanation = (
        "I examined the relationship between student-teacher ratios and academic performance "
        "using the California K-6 and K-8 district data. I constructed the student-teacher ratio "
        "as total enrollment divided by the number of teachers, and measured academic performance "
        "as the average of the district's reading and math scores.\n\n"
        f"The Pearson correlation between the student-teacher ratio and average test score was "
        f"{corr:.3f} with p-value {p_corr:.4f}, indicating {strength_desc} where {direction}, and this "
        f"pattern is {sig_desc}. A simple linear regression of average test scores on the student-teacher "
        f"ratio produced a slope of {coef:.3f} score points per one additional student per teacher "
        f"(p-value {p_coef:.4f}), with an R-squared of {r2:.3f}, meaning the ratio alone explains about "
        f"{r2 * 100:.1f}% of the cross-district variation in test scores.\n\n"
        "Taken together, these results "
    )

    if likert > 50:
        explanation += (
            "provide overall support for the claim that lower student-teacher ratios are associated "
            "with higher academic performance. "
        )
    elif likert == 50:
        explanation += (
            "do not provide clear evidence for or against a relationship between student-teacher ratios "
            "and academic performance. "
        )
    else:
        explanation += (
            "suggest that the data do not support the claim that lower student-teacher ratios are associated "
            "with higher academic performance, and in fact lean in the opposite direction. "
        )

    explanation += (
        f"On a 0–100 scale, where 0 represents a strong 'No' and 100 represents a strong 'Yes' answer to the "
        f"research question, I place the strength of the evidence at {likert}."
    )

    return explanation


def main() -> None:
    df = pd.read_csv("caschools.csv")

    ratio = compute_student_teacher_ratio(df)
    performance = compute_academic_performance(df)

    corr, p_corr, coef, p_coef, r2 = summarize_association(ratio, performance)
    likert = likert_score_from_association(corr, p_corr)
    explanation = build_explanation(corr, p_corr, coef, p_coef, r2, likert)

    result = {"response": int(likert), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
