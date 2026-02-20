import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute student-teacher ratio: enrollment / number of teachers
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    # Overall academic performance as the average of reading and math scores
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0
    # Socioeconomic and demographic controls
    df = df[
        [
            "student_teacher_ratio",
            "avg_score",
            "feature8",   # % CalWorks
            "feature9",   # % reduced-price lunch
            "feature11",  # expenditure per student
            "feature12",  # average income (thousands)
            "feature13",  # % English learners
        ]
    ].dropna()
    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    # Simple bivariate correlation
    corr = df["student_teacher_ratio"].corr(df["avg_score"])

    # Simple linear regression: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    pval_simple = float(model_simple.pvalues["student_teacher_ratio"])

    # Multiple regression with key controls
    X_controls = df[
        [
            "student_teacher_ratio",
            "feature8",
            "feature9",
            "feature11",
            "feature12",
            "feature13",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df["avg_score"], X_controls).fit()
    coef_controls = float(model_controls.params["student_teacher_ratio"])
    pval_controls = float(model_controls.pvalues["student_teacher_ratio"])

    # Decide Yes/No based on direction and statistical significance
    alpha = 0.05
    associated = (
        (corr < 0)
        and (coef_simple < 0)
        and (coef_controls < 0)
        and (pval_simple < alpha)
        and (pval_controls < alpha)
    )

    response = "Yes" if associated else "No"

    # Build human-readable explanation
    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?\n\n"
        "Data and variables:\n"
        "- The dataset covers 420 K-6 and K-8 California school districts.\n"
        "- Student-teacher ratio was computed as total enrollment (feature6) divided by the number of teachers (feature7).\n"
        "- Academic performance was measured as the average of district-level reading and math scores (features 14 and 15).\n\n"
        "Analysis:\n"
        f"- The Pearson correlation between student-teacher ratio and average test score is {corr:.3f}, indicating that scores tend to "
        f"{'decrease' if corr < 0 else 'increase' if corr > 0 else 'not change materially'} as the ratio increases.\n"
        f"- In a simple linear regression of average score on the student-teacher ratio, the coefficient on the ratio is {coef_simple:.3f} "
        f"with p-value {pval_simple:.4f}.\n"
        f"- In a multiple regression controlling for socio-economic and demographic factors (CalWorks, reduced-price lunch, expenditure per "
        f"student, income, and percent English learners), the coefficient on the student-teacher ratio is {coef_controls:.3f} with p-value "
        f"{pval_controls:.4f}.\n\n"
        "Interpretation:\n"
    )

    if associated:
        explanation += (
            "Across districts, higher student-teacher ratios are significantly associated with lower average test scores, even after "
            "controlling for key socio-economic and demographic variables. This implies that districts with fewer students per teacher "
            "tend to have higher academic performance on standardized tests, so the data support the claim that a lower student-teacher "
            "ratio is associated with higher academic performance."
        )
    else:
        explanation += (
            "The estimated relationship between student-teacher ratio and average test scores is not consistently negative and "
            "statistically significant across both simple and multiple regression models. Therefore, in this dataset we do not find "
            "clear evidence that lower student-teacher ratios are associated with higher academic performance once other observed factors "
            "are taken into account."
        )

    return {"response": response, "explanation": explanation}


def main() -> None:
    df = load_data(Path("caschools.csv"))
    results = analyze_relationship(df)
    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

