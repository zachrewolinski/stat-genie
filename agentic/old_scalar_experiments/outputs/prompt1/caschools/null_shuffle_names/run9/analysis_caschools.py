import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Based on info.json descriptions, "english" is total enrollment (students)
    # and "students" is the number of teachers (FTE).
    students_count = df["english"].astype(float)
    teachers_count = df["students"].astype(float)

    # Compute student-teacher ratio; drop rows with non-positive or missing counts.
    ratio = students_count / teachers_count.replace(0, np.nan)
    df = df.assign(student_teacher_ratio=ratio).dropna(subset=["student_teacher_ratio"])

    # Academic performance: average of reading and math scores across the district.
    # "district" and "expenditure" hold average reading and math scores respectively.
    df = df.assign(
        avg_score=(df["district"].astype(float) + df["expenditure"].astype(float)) / 2.0
    )

    # Basic descriptive association: Pearson correlation.
    corr = df["student_teacher_ratio"].corr(df["avg_score"])

    # Simple linear regression: avg_score ~ student_teacher_ratio
    y = df["avg_score"]
    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(y, X).fit()
    beta_ratio = float(model.params["student_teacher_ratio"])
    pval_ratio = float(model.pvalues["student_teacher_ratio"])
    r_squared = float(model.rsquared)

    # Decide Yes/No: "Yes" if higher ratios (more students per teacher)
    # are significantly associated with lower scores (negative slope, p < 0.05).
    associated = beta_ratio < 0 and pval_ratio < 0.05
    response = "Yes" if associated else "No"

    explanation = (
        "I used the 1998–1999 California K-6/K-8 district data. "
        "I treated total enrollment (column 'english') as the number of students and "
        "the number of teachers (column 'students') as full-time equivalent teachers, "
        "and computed the student-teacher ratio as students divided by teachers for each district. "
        "Academic performance was defined as the average of the district reading and math scores "
        "(columns 'district' and 'expenditure'). "
        f"The Pearson correlation between the student-teacher ratio and this average score was {corr:.3f}. "
        f"In a linear regression of average score on the student-teacher ratio, the estimated slope for the ratio "
        f"was {beta_ratio:.3f} with p-value {pval_ratio:.3f} and R-squared {r_squared:.3f}. "
        "Because the estimated slope is "
        f"{'negative and statistically significant' if associated else 'not a clear, significantly negative effect'} "
        "at the 5% level, this analysis "
        f"{'supports' if associated else 'does not provide strong evidence for'} the claim that lower student-teacher "
        "ratios are associated with higher academic performance across districts in this dataset."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

