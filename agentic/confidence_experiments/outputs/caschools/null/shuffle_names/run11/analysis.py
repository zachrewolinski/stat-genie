import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (for context in the explanation, even though we hard-code columns below).
    with (base_path / "info.json").open("r") as f:
        info = json.load(f)

    df = pd.read_csv(base_path / "caschools.csv")

    # Map shuffled column names to their semantic roles using the descriptions in info.json.
    # From info.json:
    #   english   -> "Total enrollment."
    #   students  -> "Number of teachers."
    #   district  -> "Average reading score."
    #   expenditure -> "Average math score."
    #   income    -> "District average income (in USD 1,000)."
    #   school    -> "Percent qualifying for CalWorks (income assistance)."
    #   computer  -> "Percent qualifying for reduced-price lunch."
    #   rownames  -> "Percent of English learners."
    #   grades    -> "Expenditure per student."
    enrollment_col = "english"
    teachers_col = "students"
    read_score_col = "district"
    math_score_col = "expenditure"
    income_col = "income"
    calworks_pct_col = "school"
    lunch_pct_col = "computer"
    ell_pct_col = "rownames"
    exp_per_student_col = "grades"

    data = df.copy()
    # Student–teacher ratio: students per teacher at the district level.
    data["stratio"] = data[enrollment_col] / data[teachers_col]
    # Academic performance: average of reading and math scores.
    data["testscr"] = (data[read_score_col] + data[math_score_col]) / 2.0

    cols_for_model = [
        "testscr",
        "stratio",
        income_col,
        calworks_pct_col,
        lunch_pct_col,
        ell_pct_col,
        exp_per_student_col,
    ]
    model_data = data[cols_for_model].replace([np.inf, -np.inf], np.nan).dropna()

    # Simple bivariate relationship: testscr ~ stratio
    X_simple = sm.add_constant(model_data["stratio"])
    model_simple = sm.OLS(model_data["testscr"], X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key covariates:
    # testscr ~ stratio + income + calworks% + lunch% + ell% + expenditure_per_student
    predictors = [
        "stratio",
        income_col,
        calworks_pct_col,
        lunch_pct_col,
        ell_pct_col,
        exp_per_student_col,
    ]
    X_full = sm.add_constant(model_data[predictors])
    model_full = sm.OLS(model_data["testscr"], X_full).fit()
    coef_full = float(model_full.params["stratio"])
    pval_full = float(model_full.pvalues["stratio"])
    r2_full = float(model_full.rsquared)

    corr = float(model_data["stratio"].corr(model_data["testscr"]))
    testscr_std = float(model_data["testscr"].std())
    # Effect on test score of reducing the student–teacher ratio by 5 students per teacher.
    effect_per_5_students = -5.0 * coef_full

    # Decide on Likert-scale strength based on sign, significance, and magnitude.
    # Start from an agnostic midpoint and adjust.
    score = 50

    if coef_full < 0:
        score += 15  # association in the hypothesized direction
    else:
        score -= 15

    # Statistical significance
    if pval_full < 0.01:
        score += 20
    elif pval_full < 0.05:
        score += 10
    elif pval_full > 0.2:
        score -= 10

    # Effect size relative to variability in test scores
    if abs(effect_per_5_students) > 0.1 * testscr_std:
        score += 10
    elif abs(effect_per_5_students) < 0.02 * testscr_std:
        score -= 5

    # Strength of simple linear relationship (R^2 and correlation)
    if r2_simple > 0.15 and abs(corr) > 0.3:
        score += 5
    elif r2_simple < 0.02 or abs(corr) < 0.05:
        score -= 5

    # Clamp to [0, 100] and convert to int
    score_int = int(round(min(100, max(0, score))))

    # Build explanation text summarizing the key evidence.
    research_question = info["research_questions"][0]

    direction_text = (
        "a lower student–teacher ratio is associated with higher average test scores"
        if coef_full < 0
        else "a lower student–teacher ratio is associated with lower average test scores"
    )

    significance_text = (
        f"highly statistically significant (p-value ≈ {pval_full:.3g})"
        if pval_full < 0.01
        else (
            f"statistically significant (p-value ≈ {pval_full:.3g})"
            if pval_full < 0.05
            else f"not strongly statistically significant (p-value ≈ {pval_full:.3g})"
        )
    )

    explanation = (
        f"Research question: {research_question} "
        f"Using data on 420 California K-6 and K-8 school districts, I constructed the "
        f"student–teacher ratio as total enrollment divided by the number of teachers and "
        f"measured academic performance as the average of the reading and math test scores. "
        f"In a simple linear regression of average test score on the student–teacher ratio, the "
        f"estimated coefficient on the ratio is {coef_simple:.3f}, with a p-value of "
        f"{pval_simple:.3g} and R² of {r2_simple:.3f}, indicating that {direction_text} in the "
        f"bivariate relationship. In a multiple regression that controls for district income, "
        f"shares of students in CalWorks and reduced-price lunch, the share of English learners, "
        f"and expenditure per student, the coefficient on the student–teacher ratio remains "
        f"{coef_full:.3f} and {significance_text}. This implies that reducing the student–teacher "
        f"ratio by 5 students per teacher is associated with an average test score change of "
        f"{effect_per_5_students:.2f} points. The correlation between the student–teacher ratio "
        f"and average test scores is {corr:.3f}, and the full model explains about {r2_full:.3f} "
        f"of the variance in test scores. Taken together, these results provide "
        f"{'strong' if score_int >= 70 else 'limited'} statistical evidence that districts with "
        f"lower student–teacher ratios tend to have higher academic performance, "
        f"which is reflected in the Likert-scale response of {score_int}."
    )

    conclusion = {"response": score_int, "explanation": explanation}

    # Write required JSON object to conclusion.txt with no extra text.
    with (base_path / "conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

