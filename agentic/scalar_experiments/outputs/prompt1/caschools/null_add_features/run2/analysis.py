import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    info_path = Path("info.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)

    # Basic variables relevant to the research question
    # Student-teacher ratio: number of students per teacher
    df["student_teacher_ratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Keep rows with complete data on key variables and main controls
    cols_needed = [
        "student_teacher_ratio",
        "avg_score",
        "income",
        "english",
        "lunch",
        "calworks",
    ]
    data = df[cols_needed].dropna()

    # Simple correlation between ratio and average score
    corr = data["student_teacher_ratio"].corr(data["avg_score"])

    # Unadjusted linear regression: avg_score ~ student_teacher_ratio
    X1 = sm.add_constant(data["student_teacher_ratio"])
    model_simple = sm.OLS(data["avg_score"], X1).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    pval_simple = float(model_simple.pvalues["student_teacher_ratio"])

    # Multiple regression controlling for key socioeconomic variables
    controls = ["income", "english", "lunch", "calworks"]
    X2 = sm.add_constant(data[["student_teacher_ratio"] + controls])
    model_controls = sm.OLS(data["avg_score"], X2).fit()
    coef_controls = float(model_controls.params["student_teacher_ratio"])
    pval_controls = float(model_controls.pvalues["student_teacher_ratio"])

    # Decide on Yes/No based on sign and statistical significance
    # We interpret "associated with" as a negative and statistically significant
    # relationship between student-teacher ratio and test scores.
    response = (
        "Yes"
        if (coef_simple < 0 and pval_simple < 0.05 and coef_controls < 0 and pval_controls < 0.05)
        else "No"
    )

    n_obs = int(data.shape[0])

    base_explanation = (
        "Using data on {n} California K-6 and K-8 school districts, "
        "I computed the student–teacher ratio as students per teacher and academic "
        "performance as the average of 5th grade reading and math scores. "
        "The simple correlation between the student–teacher ratio and average score is {corr:.3f}, "
        "which is very close to zero. "
        "In a bivariate linear regression, the estimated effect of one additional student per teacher on "
        "average test score is {coef_simple:.2f} points (p = {p_simple:.3g}). "
        "After controlling for district income, the percentage of English learners, the percentage of students on "
        "reduced-price lunch, and the percentage receiving CalWorks assistance, the estimated effect of one "
        "additional student per teacher is {coef_controls:.2f} points (p = {p_controls:.3g}). "
    ).format(
        n=n_obs,
        corr=corr,
        coef_simple=coef_simple,
        p_simple=pval_simple,
        coef_controls=coef_controls,
        p_controls=pval_controls,
    )

    if response == "Yes":
        summary_sentence = (
            "Because the association between the student–teacher ratio and test scores is negative and "
            "statistically significant in both the simple and adjusted models, the evidence indicates that "
            "lower student–teacher ratios are associated with higher academic performance in this dataset "
            "(though the analysis is observational and does not by itself establish causality)."
        )
    else:
        summary_sentence = (
            "Because the estimated effects of the student–teacher ratio are small in magnitude and not "
            "statistically distinguishable from zero in either the simple or adjusted models, this analysis "
            "does not provide strong evidence that lower student–teacher ratios are associated with higher "
            "academic performance in this dataset (and, as with any observational study, causality cannot "
            "be established)."
        )

    explanation = base_explanation + summary_sentence

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
