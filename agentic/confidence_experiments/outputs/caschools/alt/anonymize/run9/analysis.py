import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop any rows with missing values in the variables we use
    cols_for_simple = ["student_teacher_ratio", "avg_score"]
    cols_for_controls = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    df_simple = df[cols_for_simple].dropna()
    df_multi = df[["avg_score", "student_teacher_ratio"] + cols_for_controls].dropna()

    # Correlation between student-teacher ratio and average score
    corr, p_corr = stats.pearsonr(
        df_simple["student_teacher_ratio"], df_simple["avg_score"]
    )

    # Simple OLS: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df_simple["student_teacher_ratio"])
    model_simple = sm.OLS(df_simple["avg_score"], X_simple).fit()
    beta_str_simple = float(model_simple.params["student_teacher_ratio"])
    p_str_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple OLS with key socioeconomic controls
    X_multi = sm.add_constant(
        df_multi[["student_teacher_ratio"] + cols_for_controls]
    )
    model_multi = sm.OLS(df_multi["avg_score"], X_multi).fit()
    beta_str_multi = float(model_multi.params["student_teacher_ratio"])
    p_str_multi = float(model_multi.pvalues["student_teacher_ratio"])
    r2_multi = float(model_multi.rsquared)

    # Effect size: change in avg_score for a 5-student reduction in ratio
    # (negative coefficient means higher ratio is associated with lower scores)
    effect_5_students = -5.0 * beta_str_multi

    # Map statistical evidence to Likert-style 0-100 scale
    # Default neutral value
    response_value = 50

    # Strong, consistent negative association with high significance
    if beta_str_multi < 0 and p_str_multi < 0.001 and abs(effect_5_students) >= 5:
        response_value = 90
    # Clear negative association with conventional significance
    elif beta_str_multi < 0 and p_str_multi < 0.05:
        response_value = 80
    # Some evidence but weaker or marginally significant
    elif beta_str_multi < 0 and p_str_multi < 0.1:
        response_value = 65
    # No clear evidence of the hypothesized relationship
    elif p_str_multi >= 0.1 or beta_str_multi >= 0:
        response_value = 30

    # Build explanation text using the actual estimates
    explanation_parts = []
    explanation_parts.append(
        f"Using data on {len(df)} California K-6 and K-8 school districts, "
        f"I examined whether a lower student-teacher ratio is associated with higher academic performance "
        "measured as the average of district-level 5th-grade reading and math scores."
    )
    explanation_parts.append(
        f"The simple Pearson correlation between the student-teacher ratio and average test score "
        f"is {corr:.3f} with a p-value of {p_corr:.3g}, indicating that districts with more students per teacher "
        f"tend to have "
        + ("lower" if corr < 0 else "higher")
        + " test scores."
    )
    explanation_parts.append(
        f"In a simple linear regression of average test scores on the student-teacher ratio, "
        f"the estimated coefficient on the ratio is {beta_str_simple:.3f} (p = {p_str_simple:.3g}), "
        f"with an R-squared of {r2_simple:.3f}."
    )
    explanation_parts.append(
        "To account for potential confounding factors, I next estimated a multiple regression that includes "
        "controls for the percent of students on income assistance (CalWorks), the percent eligible for "
        "reduced-price lunch, expenditure per student, average district income, and the percent of English learners."
    )
    explanation_parts.append(
        f"In this multiple regression, the coefficient on the student-teacher ratio is {beta_str_multi:.3f} "
        f"(p = {p_str_multi:.3g}), with an R-squared of {r2_multi:.3f}."
    )
    explanation_parts.append(
        f"Interpreting the multiple-regression estimate, a reduction of 5 students per teacher is associated "
        f"with an average change of about {effect_5_students:.2f} points in the test score measure, holding the "
        f"other demographic and resource variables constant."
    )

    if beta_str_multi < 0 and p_str_multi < 0.05:
        interpretation = (
            "Both the correlation and regression results show a negative and statistically significant relationship "
            "between the student-teacher ratio and test scores, suggesting that districts with smaller classes "
            "tend to perform better academically."
        )
    elif beta_str_multi < 0 and p_str_multi < 0.1:
        interpretation = (
            "The estimated relationship between the student-teacher ratio and test scores is negative and marginally "
            "statistically significant after controlling for demographics, providing some but not overwhelming evidence "
            "that smaller classes are associated with better performance."
        )
    elif beta_str_multi < 0:
        interpretation = (
            "Although the estimated relationship between the student-teacher ratio and test scores is negative, "
            "it is not statistically distinguishable from zero in the multiple regression, so the evidence for a real "
            "association is weak."
        )
    else:
        interpretation = (
            "The estimated relationship between the student-teacher ratio and test scores is not negative once "
            "controls are included, and it is not statistically significant, indicating little evidence that smaller "
            "classes are associated with higher performance in this dataset."
        )

    explanation_parts.append(interpretation)
    explanation_parts.append(
        f"On a 0–100 scale where 0 represents a strong 'No' and 100 a strong 'Yes', "
        f"I summarize this evidence with a score of {response_value}, reflecting the strength and statistical "
        f"reliability of the observed association between smaller student-teacher ratios and higher academic performance."
    )

    explanation = " ".join(explanation_parts)

    result = {"response": int(response_value), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
