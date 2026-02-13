import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json metadata:
    # - "english" column: total enrollment (students)
    # - "students" column: number of teachers
    # - "district": average reading score
    # - "expenditure": average math score
    # - "income": district average income (in USD 1,000)
    # - "school": percent qualifying for CalWorks (income assistance)
    # - "computer": percent qualifying for reduced-price lunch
    # - "rownames": percent of English learners
    # - "grades": expenditure per student

    # Construct key variables
    df = df.copy()
    df["stu_teacher_ratio"] = df["english"] / df["students"]
    df["avg_test_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any rows with missing or non-finite values
    subset = df[["stu_teacher_ratio", "avg_test_score", "income", "school", "computer", "rownames", "grades"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Simple Pearson correlation between ratio and average test score
    corr, corr_p = pearsonr(subset["stu_teacher_ratio"], subset["avg_test_score"])

    # Simple linear regression: avg_test_score ~ stu_teacher_ratio
    X_simple = sm.add_constant(subset["stu_teacher_ratio"])
    model_simple = sm.OLS(subset["avg_test_score"], X_simple).fit()
    coef_simple = float(model_simple.params["stu_teacher_ratio"])
    pval_simple = float(model_simple.pvalues["stu_teacher_ratio"])

    # Multiple regression with basic demographic and spending controls
    controls = ["income", "school", "computer", "rownames", "grades"]
    X_full = sm.add_constant(subset[["stu_teacher_ratio"] + controls])
    model_full = sm.OLS(subset["avg_test_score"], X_full).fit()
    coef_full = float(model_full.params["stu_teacher_ratio"])
    pval_full = float(model_full.pvalues["stu_teacher_ratio"])

    # Decision rule:
    # Answer "Yes" if the student–teacher ratio is negatively associated
    # with test scores and the association remains statistically significant
    # (p < 0.05) in the regression with controls.
    negative_and_significant = (coef_full < 0) and (pval_full < 0.05)
    response = "Yes" if negative_and_significant else "No"

    # Build a concise explanation summarizing the key evidence
    common_part = (
        "I constructed a student–teacher ratio as total enrollment divided by the number of teachers, "
        "and an academic performance measure as the average of district reading and math scores. "
        f"The Pearson correlation between the ratio and average test score was {corr:.3f} (p-value {corr_p:.3g}), "
        "summarizing the direction and strength of the bivariate association. "
        "I then estimated an ordinary least squares regression of average test scores on the student–teacher ratio, "
        "first without controls and then controlling for district income, shares of low-income and English-learner students, "
        "and per-pupil expenditure. "
        f"In the model with controls, the coefficient on the student–teacher ratio was {coef_full:.3f} "
        f"with p-value {pval_full:.3g}. "
    )

    if negative_and_significant:
        tail = (
            "This negative and statistically significant coefficient implies that, holding these factors fixed, "
            "an increase in the number of students per teacher is associated with a decrease in average test scores. "
            "Because lower student–teacher ratios correspond to fewer students per teacher, the data provide evidence "
            "that districts with lower student–teacher ratios tend to have higher academic performance."
        )
    else:
        tail = (
            "This coefficient is small in magnitude and not statistically distinguishable from zero at conventional levels, "
            "so, after accounting for these demographic and spending factors, the data do not provide clear evidence that "
            "student–teacher ratios are systematically related to average test scores in this sample."
        )

    explanation = common_part + tail

    result = {"response": response, "explanation": explanation}

    # Write required JSON output to conclusion.txt
    output_path = Path("conclusion.txt")
    output_path.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
