import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata (not strictly necessary for analysis, but documents context)
    info_path = Path("info.json")
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    # Load dataset
    df = pd.read_csv("caschools.csv")

    # According to the metadata, feature6 is total enrollment and feature7 is number of teachers.
    # Construct student-teacher ratio as students per teacher.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Use average reading and math scores (features 14 and 15) as academic performance.
    df["avg_testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop rows with missing values in the key variables (should be none, but safe).
    data = df[["student_teacher_ratio", "avg_testscr", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13"]].dropna()

    # Basic correlation between student-teacher ratio and test scores
    corr = data["avg_testscr"].corr(data["student_teacher_ratio"])

    # Simple bivariate OLS regression: test score on student-teacher ratio
    X_simple = sm.add_constant(data[["student_teacher_ratio"]])
    y = data["avg_testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    coef_simple = model_simple.params["student_teacher_ratio"]
    pval_simple = model_simple.pvalues["student_teacher_ratio"]

    # Multivariate regression controlling for other observed school/demographic characteristics.
    # We treat features 8-13 as generic controls (percent disadvantaged students, computers per student,
    # expenditures, etc., per the dataset description).
    control_cols = ["feature8", "feature9", "feature10", "feature11", "feature12", "feature13"]
    X_full = sm.add_constant(data[["student_teacher_ratio"] + control_cols])
    model_full = sm.OLS(y, X_full).fit()

    coef_full = model_full.params["student_teacher_ratio"]
    pval_full = model_full.pvalues["student_teacher_ratio"]

    # Decide on Yes/No based on direction and significance of the association.
    # Lower ratios correspond to smaller numeric values of student_teacher_ratio.
    # A negative coefficient means higher ratios (more students per teacher) are associated with lower scores,
    # equivalently, lower ratios are associated with higher scores.
    associated = coef_full < 0 and pval_full < 0.05

    response = "Yes" if associated else "No"

    # Build explanation text.
    explanation_lines = []
    explanation_lines.append(
        "I computed a student-teacher ratio for each of the 420 California K-6/K-8 districts as total "
        "enrollment (feature6) divided by the number of teachers (feature7), and defined academic "
        "performance as the average of the district's mean reading and math scores (features 14 and 15)."
    )
    direction = "lower" if corr < 0 else "higher" if corr > 0 else "similar"
    explanation_lines.append(
        f"Across districts, the Pearson correlation between student-teacher ratio and average test score "
        f"was {corr:.3f}, indicating that districts with more students per teacher tend to have {direction} "
        f"test scores."
    )
    explanation_lines.append(
        f"In a simple linear regression of test scores on the student-teacher ratio, the estimated "
        f"coefficient on the ratio was {coef_simple:.3f} with a p-value of {pval_simple:.3g}."
    )
    explanation_lines.append(
        f"In a regression that additionally controlled for other observed district characteristics (features 8–13, "
        f"capturing factors such as student demographics and resources), the coefficient on the student-teacher "
        f"ratio remained {coef_full:.3f} with a p-value of {pval_full:.3g}."
    )
    if associated:
        explanation_lines.append(
            "Because the estimated coefficient on the student-teacher ratio is negative and statistically "
            "significant even after adjusting for these controls, the data show that districts with lower "
            "student-teacher ratios tend to have higher average test scores. This provides evidence of an "
            "association between smaller student-teacher ratios and better academic performance, though the "
            "observational design does not by itself prove causality."
        )
    else:
        explanation_lines.append(
            "Because the estimated coefficient on the student-teacher ratio is not consistently negative and "
            "statistically significant once other district characteristics are taken into account, the data do "
            "not provide clear evidence that lower student-teacher ratios are associated with higher academic "
            "performance in this sample."
        )

    explanation = " " .join(explanation_lines)

    # Write required JSON output to conclusion.txt
    out = {"response": response, "explanation": explanation}
    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
