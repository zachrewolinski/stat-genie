import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata
    info_path = Path("info.json")
    with info_path.open("r") as f:
        info = json.load(f)

    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on the documented structure
    # feature6: total enrollment (students), feature7: number of teachers
    # feature14: average reading score, feature15: average math score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop obviously problematic rows (e.g., missing or infinite ratios)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["student_teacher_ratio", "test_score"]
    )

    # Simple correlation
    corr = df["student_teacher_ratio"].corr(df["test_score"])

    # Simple linear regression: test_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["test_score"], X_simple).fit()
    coef_simple = model_simple.params["student_teacher_ratio"]
    pval_simple = model_simple.pvalues["student_teacher_ratio"]

    # Multiple regression with key controls commonly used with this dataset
    # feature8: pct CalWorks, feature9: pct reduced-price lunch,
    # feature11: expenditure per student, feature12: district avg income,
    # feature13: pct English learners.
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    available_controls = [c for c in controls if c in df.columns]
    X_controls = df[["student_teacher_ratio"] + available_controls]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df["test_score"], X_controls).fit()
    coef_controls = model_controls.params["student_teacher_ratio"]
    pval_controls = model_controls.pvalues["student_teacher_ratio"]

    # Decide on Yes/No based on direction and significance of association.
    # We treat evidence as supportive if the coefficient is negative and
    # statistically significant at the 5% level in both simple and controlled models.
    negative_and_sig_simple = (coef_simple < 0) and (pval_simple < 0.05)
    negative_and_sig_controls = (coef_controls < 0) and (pval_controls < 0.05)

    if negative_and_sig_simple and negative_and_sig_controls:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string summarizing the key evidence in plain language.
    explanation = (
        "Using data on 420 California K-6 and K-8 school districts, "
        "I constructed the student–teacher ratio as total enrollment divided by the "
        "number of teachers and an overall academic performance measure as the "
        "average of district reading and math test scores. "
        f"The simple correlation between the student–teacher ratio and test scores is {corr:.3f}, "
        "indicating how test scores co-vary with class size. "
        f"In a simple linear regression of test scores on the student–teacher ratio, the estimated "
        f"coefficient on the ratio is {coef_simple:.3f} with a p-value of {pval_simple:.3f}. "
        f"When adding controls for student demographics and resources (CalWorks share, reduced-price lunch share, "
        f"expenditure per student, average income, and percent English learners), the coefficient on the "
        f"student–teacher ratio becomes {coef_controls:.3f} with a p-value of {pval_controls:.3f}. "
        "These estimates indicate whether districts with lower student–teacher ratios tend to have higher test scores, "
        "and the statistical significance levels show how strong and reliable this association is after accounting "
        "for observable differences across districts."
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write the required output file with only the JSON object.
    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

