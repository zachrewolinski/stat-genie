import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map shuffled column names to their semantic meaning using info.json:
    # english    -> total enrollment
    # students   -> number of teachers
    # district   -> average reading score
    # expenditure -> average math score
    df = df.copy()
    df["student_teacher_ratio"] = df["english"] / df["students"]
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    analysis_df = df[["student_teacher_ratio", "avg_score"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    x = analysis_df["student_teacher_ratio"]
    y = analysis_df["avg_score"]

    # Correlation between student–teacher ratio and academic performance.
    r, p_corr = stats.pearsonr(x, y)

    # Simple linear regression: avg_score ~ student_teacher_ratio
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = float(model.params["student_teacher_ratio"])
    p_slope = float(model.pvalues["student_teacher_ratio"])
    r_squared = float(model.rsquared)

    # Determine Likert-scale response (0–100) reflecting strength of evidence
    # that lower student–teacher ratios are associated with higher performance.
    evidence_score = compute_likert_score(slope, p_slope, r)

    direction = "lower" if slope < 0 else "higher"
    explanation = build_explanation(
        slope=slope,
        p_slope=p_slope,
        r=r,
        r_squared=r_squared,
        direction=direction,
        response=evidence_score,
        n=len(analysis_df),
    )

    conclusion = {"response": int(evidence_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def compute_likert_score(slope: float, p_value: float, r: float) -> int:
    """
    Map the regression results to a 0–100 Likert-style confidence score.

    The score emphasizes:
    - statistical significance of the slope
    - effect size via |r|
    - direction: we are answering whether *lower* ratios are associated with
      *higher* performance, which corresponds to a negative slope.
    """
    # Start from a neutral baseline.
    score = 50.0

    # Direction: negative slope supports the research hypothesis.
    if slope < 0:
        score += 10.0
    else:
        score -= 10.0

    # Statistical significance of the slope.
    if p_value < 1e-3:
        score += 25.0
    elif p_value < 1e-2:
        score += 20.0
    elif p_value < 5e-2:
        score += 10.0
    elif p_value < 0.1:
        score += 5.0
    else:
        score -= 10.0

    # Effect size via absolute correlation.
    abs_r = abs(r)
    if abs_r >= 0.5:
        score += 15.0
    elif abs_r >= 0.3:
        score += 10.0
    elif abs_r >= 0.1:
        score += 5.0
    else:
        score -= 5.0

    # Clamp to [0, 100] and return as integer.
    score = max(0.0, min(100.0, score))
    return int(round(score))


def build_explanation(
    slope: float,
    p_slope: float,
    r: float,
    r_squared: float,
    direction: str,
    response: int,
    n: int,
) -> str:
    """
    Construct a human-readable explanation summarizing the analysis and results.
    """
    relationship = (
        "a lower student–teacher ratio is associated with higher test scores"
        if slope < 0
        else "a higher student–teacher ratio is associated with higher test scores"
    )

    lines = [
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?",
        "Data: 420 California K-6/K-8 districts with 5th-grade Stanford 9 test scores and district characteristics.",
        "Operationalization:",
        "- Student–teacher ratio = total enrollment (column 'english') divided by number of teachers (column 'students').",
        "- Academic performance = average of reading and math scores (columns 'district' and 'expenditure').",
        f"Sample size used after removing missing values: n = {n}.",
        "Analysis:",
        "- Computed the Pearson correlation between student–teacher ratio and average test score.",
        "- Fit a simple linear regression: average test score ~ student–teacher ratio.",
        f"Key results:",
        f"- Pearson correlation r = {r:.3f}.",
        f"- Regression slope for student–teacher ratio = {slope:.3f} score points per additional student per teacher (p = {p_slope:.3g}).",
        f"- R-squared of the regression model = {r_squared:.3f}.",
        f"Interpretation: In this sample, {relationship}. Each additional student per teacher is associated with an average change of {slope:.2f} points in the combined test score.",
        f"Conclusion: On a 0–100 scale, the evidence that lower student–teacher ratios are associated with higher academic performance is summarized by a score of {response}.",
    ]

    return " ".join(lines)


if __name__ == "__main__":
    main()

