import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on metadata in info.json
    # feature6: total enrollment
    # feature7: number of teachers
    # feature14: average reading score
    # feature15: average math score
    df = df.copy()
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Keep only rows with non-missing values for variables used in analysis
    cols_needed = [
        "student_teacher_ratio",
        "testscr",
        "feature8",   # CalWorks %
        "feature9",   # Reduced-price lunch %
        "feature10",  # Number of computers
        "feature11",  # Expenditure per student
        "feature12",  # District average income
        "feature13",  # Percent English learners
    ]
    df_model = df[cols_needed].dropna()

    # Correlation between student-teacher ratio and test scores
    str_values = df_model["student_teacher_ratio"].to_numpy()
    testscr_values = df_model["testscr"].to_numpy()
    corr, corr_p = stats.pearsonr(str_values, testscr_values)

    # Simple linear regression: testscr ~ student_teacher_ratio
    X_simple = sm.add_constant(df_model["student_teacher_ratio"])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression with key demographic and resource controls
    X_controls = df_model[
        [
            "student_teacher_ratio",
            "feature8",
            "feature9",
            "feature10",
            "feature11",
            "feature12",
            "feature13",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(y, X_controls).fit()

    coef_str_simple = float(model_simple.params["student_teacher_ratio"])
    p_str_simple = float(model_simple.pvalues["student_teacher_ratio"])
    coef_str_controls = float(model_controls.params["student_teacher_ratio"])
    p_str_controls = float(model_controls.pvalues["student_teacher_ratio"])

    r2_simple = float(model_simple.rsquared)
    r2_controls = float(model_controls.rsquared)

    # Map statistical evidence to a 0-100 Likert-style confidence that
    # "lower student-teacher ratio is associated with higher academic performance".
    # Negative coefficients and correlations (higher ratio -> lower scores)
    # support a "Yes" answer.
    def compute_response(
        corr_value: float, p_value: float, coef_value: float
    ) -> int:
        if coef_value < 0 and p_value < 0.001 and abs(corr_value) >= 0.3:
            return 90
        if coef_value < 0 and p_value < 0.01 and abs(corr_value) >= 0.2:
            return 80
        if coef_value < 0 and p_value < 0.05 and abs(corr_value) >= 0.1:
            return 70
        if coef_value < 0 and p_value < 0.1:
            return 60
        if coef_value < 0:
            return 55
        # Evidence does not clearly support the claim
        if p_value > 0.1 or abs(corr_value) < 0.1:
            return 40
        return 50

    response = compute_response(corr, p_str_controls, coef_str_controls)

    if coef_str_controls < 0 and p_str_controls < 0.05:
        evidence_phrase = "supports"
    elif coef_str_controls < 0:
        evidence_phrase = "provides only weak support for"
    else:
        evidence_phrase = "does not provide clear support for"

    if response >= 80:
        relation_desc = "moderately strong"
    elif response >= 60:
        relation_desc = "modest"
    else:
        relation_desc = "weak or negligible"

    explanation_lines = [
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?",
        (
            "Using the caschools.csv dataset of 420 California K-6/K-8 districts, "
            "I constructed a student-teacher ratio as total enrollment (feature6) divided by number of teachers "
            "(feature7), and an academic performance measure as the average of district reading and math scores "
            "(features14 and 15)."
        ),
        (
            f"The Pearson correlation between student-teacher ratio and test scores is {corr:.3f} "
            f"(p = {corr_p:.3g}), indicating that districts with more students per teacher tend to have "
            f"{'lower' if corr < 0 else 'higher'} average test scores."
        ),
        (
            f"A simple linear regression of test scores on student-teacher ratio yields a coefficient of "
            f"{coef_str_simple:.3f} (p = {p_str_simple:.3g}, R^2 = {r2_simple:.3f}). "
            "This coefficient is interpreted as the expected change in average test scores for a one-unit "
            "increase in the student-teacher ratio."
        ),
        (
            "To adjust for observable differences across districts, I estimated a multiple regression including "
            "controls for CalWorks participation (feature8), reduced-price lunch (feature9), number of computers "
            "(feature10), expenditure per student (feature11), district average income (feature12), and the "
            "percentage of English learners (feature13)."
        ),
        (
            f"In this controlled model, the coefficient on student-teacher ratio is {coef_str_controls:.3f} "
            f"(p = {p_str_controls:.3g}, R^2 = {r2_controls:.3f}). "
            f"The coefficient remains {'negative' if coef_str_controls < 0 else 'positive'} after accounting for "
            "these covariates, which "
            f"{evidence_phrase} the claim that lower student-teacher ratios are associated with higher academic "
            "performance."
        ),
        (
            "Overall, the direction and statistical significance of the association suggest a "
            f"{relation_desc} relationship, but the observational nature of "
            "the data means that unmeasured confounding factors may still influence the results. The Likert-style "
            "response above summarizes my level of confidence that the association is real and positive "
            "(i.e., lower ratios correspond to better outcomes)."
        ),
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
