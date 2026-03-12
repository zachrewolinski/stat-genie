import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Student-teacher ratio: total enrollment / number of teachers.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop rows with missing or non-finite values.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["student_teacher_ratio", "avg_score"]
    )

    # Basic descriptive statistics.
    ratio_desc = df["student_teacher_ratio"].describe()
    score_desc = df["avg_score"].describe()

    # Correlation between ratio and performance.
    corr = df["student_teacher_ratio"].corr(df["avg_score"])

    # Simple OLS regression: avg_score ~ student_teacher_ratio.
    X = sm.add_constant(df["student_teacher_ratio"])
    y = df["avg_score"]
    model = sm.OLS(y, X).fit()

    coef_ratio = model.params["student_teacher_ratio"]
    p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    # Decide Likert-style response based on direction, significance, and effect size.
    # Lower ratio (smaller classes) is expected to increase scores,
    # so we look for a negative coefficient.
    if p_value < 0.001 and coef_ratio < 0:
        # Strong, highly significant negative association.
        response_score = 90
        qualitative = "strong"
    elif p_value < 0.01 and coef_ratio < 0:
        response_score = 80
        qualitative = "clear"
    elif p_value < 0.05 and coef_ratio < 0:
        response_score = 70
        qualitative = "moderate"
    elif p_value < 0.05 and coef_ratio > 0:
        # Significant, but opposite to expected direction.
        response_score = 20
        qualitative = "significant but opposite-direction"
    else:
        # No strong statistical evidence of the expected association.
        response_score = 30
        qualitative = "weak or statistically uncertain"

    # Build a concise explanation summarizing evidence.
    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance? "
        "Using the caschools dataset (N = {n}), I constructed a student–teacher ratio as total enrollment "
        "divided by the number of teachers (feature6 / feature7) and an overall academic performance measure "
        "as the average of reading and math scores (features 14 and 15). "
        "The student–teacher ratio ranges from {ratio_min:.1f} to {ratio_max:.1f} "
        "(mean {ratio_mean:.1f}), while average test scores range from {score_min:.1f} to {score_max:.1f} "
        "(mean {score_mean:.1f}). "
        "The Pearson correlation between student–teacher ratio and average test score is {corr:.3f}, "
        "indicating that districts with lower ratios (smaller classes) tend to have "
        "{direction} test scores. "
        "A simple OLS regression of average test scores on the student–teacher ratio yields a coefficient of "
        "{coef_ratio:.3f} (p-value = {p_value:.3g}, R² = {r_squared:.3f}). "
        "This coefficient implies that a one-student increase in the student–teacher ratio is associated with "
        "an average change of {coef_ratio:.3f} points in test scores. "
        "Given the {qualitative} negative association and conventional levels of statistical significance, "
        "I conclude that there is {strength} evidence that lower student–teacher ratios are associated with "
        "higher academic performance in this dataset. "
        "However, this is an observational cross-sectional analysis, so the results should be interpreted as "
        "associations rather than definitive causal effects."
    ).format(
        n=len(df),
        ratio_min=ratio_desc["min"],
        ratio_max=ratio_desc["max"],
        ratio_mean=ratio_desc["mean"],
        score_min=score_desc["min"],
        score_max=score_desc["max"],
        score_mean=score_desc["mean"],
        corr=corr,
        direction="higher" if corr < 0 else "lower" if corr > 0 else "similar",
        coef_ratio=coef_ratio,
        p_value=p_value,
        r_squared=r_squared,
        qualitative=qualitative,
        strength="strong" if response_score >= 80 else "moderate" if response_score >= 60 else "limited",
    )

    output = {"response": int(response_score), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

