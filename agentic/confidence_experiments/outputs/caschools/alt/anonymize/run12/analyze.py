import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    # Student-teacher ratio: total enrollment divided by number of teachers
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop any rows with missing values in key variables (should be none, but safe)
    df_clean = df[["student_teacher_ratio", "avg_score", "feature8", "feature9", "feature11", "feature12", "feature13"]].dropna()

    # 1. Simple correlation between student-teacher ratio and average score
    r, p_value = stats.pearsonr(df_clean["student_teacher_ratio"], df_clean["avg_score"])

    # 2. Simple linear regression: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df_clean["student_teacher_ratio"])
    model_simple = sm.OLS(df_clean["avg_score"], X_simple).fit()
    coef_str = model_simple.params["student_teacher_ratio"]
    p_str = model_simple.pvalues["student_teacher_ratio"]

    # 3. Regression with basic controls for demographics and resources
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X_controls = sm.add_constant(df_clean[["student_teacher_ratio"] + controls])
    model_controls = sm.OLS(df_clean["avg_score"], X_controls).fit()
    coef_str_ctrl = model_controls.params["student_teacher_ratio"]
    p_str_ctrl = model_controls.pvalues["student_teacher_ratio"]

    # Map statistical evidence to a 0–100 Likert-style scale
    response_score = map_evidence_to_scale(
        r=r,
        p_value=p_value,
        coef_simple=coef_str,
        p_simple=p_str,
        coef_controls=coef_str_ctrl,
        p_controls=p_str_ctrl,
    )

    explanation = build_explanation(
        r=r,
        p_value=p_value,
        coef_simple=coef_str,
        p_simple=p_str,
        coef_controls=coef_str_ctrl,
        p_controls=p_str_ctrl,
        response_score=response_score,
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def map_evidence_to_scale(
    r: float,
    p_value: float,
    coef_simple: float,
    p_simple: float,
    coef_controls: float,
    p_controls: float,
) -> int:
    """
    Map the strength and consistency of evidence that lower student-teacher ratios
    are associated with higher academic performance onto a 0–100 scale.
    Higher values indicate a stronger "Yes".
    """
    # Require the association to be negative (lower ratio -> higher scores)
    negative_and_significant = (coef_simple < 0 and p_simple < 0.05) and (
        coef_controls < 0 and p_controls < 0.05
    )

    # Use correlation magnitude as an effect-size proxy
    abs_r = abs(r)

    if negative_and_significant:
        if abs_r >= 0.5 and p_value < 1e-6:
            score = 90
        elif abs_r >= 0.3 and p_value < 1e-4:
            score = 80
        elif abs_r >= 0.2 and p_value < 0.01:
            score = 70
        else:
            score = 60
    elif coef_simple < 0 or coef_controls < 0:
        # Direction is consistent but evidence is weak or marginal
        score = 45
    else:
        # No evidence that lower ratios are associated with higher performance
        score = 20

    # Ensure integer and within [0, 100]
    score = int(round(score))
    score = max(0, min(100, score))
    return score


def build_explanation(
    r: float,
    p_value: float,
    coef_simple: float,
    p_simple: float,
    coef_controls: float,
    p_controls: float,
    response_score: int,
) -> str:
    direction = "negative" if r < 0 else "positive"
    strength_desc = (
        "strong" if abs(r) >= 0.5 else "moderate" if abs(r) >= 0.3 else "weak"
    )

    yes_no_text = (
        "Yes" if response_score >= 50 else "No"
    )

    explanation = (
        f"{yes_no_text}: The data show a {direction} correlation (r = {r:.3f}, p = {p_value:.2e}) "
        f"between the student-teacher ratio and average test scores, meaning districts with fewer students "
        f"per teacher tend to have higher academic performance. "
        f"In a simple regression of average test score on the student-teacher ratio, the coefficient is "
        f"{coef_simple:.2f} (p = {p_simple:.2e}), implying that each additional student per teacher is "
        f"associated with a decrease of about {abs(coef_simple):.2f} points in test scores. "
        f"This relationship remains {('statistically significant' if p_controls < 0.05 else 'similar but not statistically significant')} "
        f"after controlling for key demographic and resource variables, with a coefficient of "
        f"{coef_controls:.2f} (p = {p_controls:.2e}). "
        f"Taken together, this provides {strength_desc} statistical evidence that lower student-teacher ratios "
        f"are associated with higher academic performance in this sample, which I summarize as a response of "
        f"{response_score} on a 0–100 scale (higher values indicating a stronger 'Yes')."
    )

    return explanation


if __name__ == "__main__":
    main()

