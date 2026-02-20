import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Derived variables
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Keep rows with complete data for variables of interest
    core_cols = [
        "student_teacher_ratio",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
        "avg_score",
    ]
    analysis_df = df[core_cols].dropna()

    ratio = analysis_df["student_teacher_ratio"]
    score = analysis_df["avg_score"]

    # Correlation analysis
    corr, corr_p = stats.pearsonr(ratio, score)

    # Simple linear regression: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(ratio)
    model_simple = sm.OLS(score, X_simple).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    p_simple = float(model_simple.pvalues["student_teacher_ratio"])

    # Multiple regression with demographic and resource controls
    controls = analysis_df[
        ["feature8", "feature9", "feature10", "feature11", "feature12", "feature13"]
    ]
    X_multi = sm.add_constant(pd.concat([ratio, controls], axis=1))
    model_multi = sm.OLS(score, X_multi).fit()
    coef_multi = float(model_multi.params["student_teacher_ratio"])
    p_multi = float(model_multi.pvalues["student_teacher_ratio"])

    # Determine direction and strength of association
    evidence_for = 0
    evidence_against = 0

    if coef_simple < 0:
        evidence_for += 1
    else:
        evidence_against += 1

    if coef_multi < 0:
        evidence_for += 1
    else:
        evidence_against += 1

    if p_simple < 0.05:
        if coef_simple < 0:
            evidence_for += 1
        else:
            evidence_against += 1

    if p_multi < 0.05:
        if coef_multi < 0:
            evidence_for += 1
        else:
            evidence_against += 1

    response = "Yes" if evidence_for >= evidence_against else "No"

    # Confidence heuristic based on consistency and significance
    confidence = 50.0

    if coef_simple < 0 and coef_multi < 0:
        confidence += 15
    elif coef_simple < 0 or coef_multi < 0:
        confidence += 5
    else:
        confidence -= 10

    # Add confidence based on statistical significance
    for p_val in (p_simple, p_multi):
        if p_val < 0.01:
            confidence += 15
        elif p_val < 0.05:
            confidence += 10
        elif p_val < 0.1:
            confidence += 5

    # Bound confidence between 0 and 100
    confidence = max(0.0, min(100.0, confidence))

    # Construct explanation string with key evidence
    explanation_parts = [
        "I computed the student-teacher ratio as total enrollment divided by number of teachers, ",
        "and measured academic performance as the average of district reading and math scores.",
        f" The Pearson correlation between the ratio and average score was {corr:.3f} (p = {corr_p:.3g}),",
        " indicating that higher ratios (more students per teacher) are "
        + ("associated with lower scores" if corr < 0 else "not clearly associated with lower scores"),
        ". In a simple linear regression of average score on the ratio, ",
        f"each additional student per teacher was associated with a {coef_simple:.2f}-point change in average score "
        f"(p = {p_simple:.3g}).",
        " In a multiple regression controlling for CalWorks participation, reduced-price lunch, computers, ",
        "expenditures, district income, and percent English learners, ",
        f"the coefficient on the student-teacher ratio was {coef_multi:.2f} (p = {p_multi:.3g}).",
    ]

    if response == "Yes":
        explanation_parts.append(
            " Across these analyses, smaller student-teacher ratios consistently predict higher academic performance, "
            "so I conclude that lower student-teacher ratios are associated with higher test scores in this dataset."
        )
    else:
        explanation_parts.append(
            " Because the association is not consistently negative and statistically meaningful across models, "
            "I do not find clear evidence that lower student-teacher ratios are associated with higher test scores in this dataset."
        )

    explanation = "".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": round(confidence, 1),
        "explanation": explanation,
    }

    # Write required JSON object to conclusion.txt with no extra text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

