import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables, if any
    key_cols = [
        "stratio",
        "testscr",
        "income",
        "calworks",
        "lunch",
        "english",
        "expenditure",
        "computer",
        "grades",
    ]
    df_model = df.dropna(subset=key_cols).copy()

    # Simple correlation between student-teacher ratio and test scores
    corr_testscr = df_model["stratio"].corr(df_model["testscr"])

    # Simple regression: test scores on student-teacher ratio
    simple_model = smf.ols("testscr ~ stratio", data=df_model).fit()
    coef_simple = float(simple_model.params["stratio"])
    pval_simple = float(simple_model.pvalues["stratio"])

    # Multiple regression controlling for observed covariates
    full_formula = (
        "testscr ~ stratio + income + calworks + lunch + english "
        "+ expenditure + computer + C(grades)"
    )
    full_model = smf.ols(full_formula, data=df_model).fit()
    coef_full = float(full_model.params["stratio"])
    pval_full = float(full_model.pvalues["stratio"])

    # Decide Yes/No based on sign and significance of the adjusted association
    if (coef_full < 0) and (pval_full < 0.05):
        response = "Yes"
    else:
        response = "No"

    # Build human-readable explanation using the main numerical results
    n_obs = int(df_model.shape[0])

    explanation = (
        "Using the CASchools dataset with "
        f"{n_obs} districts, I computed the student–teacher ratio as the number "
        "of students divided by the number of teachers and defined overall "
        "academic performance as the average of the district’s reading and math "
        "scores. The Pearson correlation between the student–teacher ratio and "
        f"this combined test score is {corr_testscr:.3f}, indicating that "
        "districts with smaller classes (lower ratios) tend to have higher test "
        "scores when the correlation is negative. In a simple linear regression "
        "of the combined test score on the student–teacher ratio, the estimated "
        f"slope is {coef_simple:.3f} points per one-student increase in the "
        f"ratio (p-value = {pval_simple:.3g}). I then fit a multiple regression "
        "controlling for average district income, the percentages of students on "
        "CalWorks and reduced-price lunch, the percentage of English learners, "
        "per-pupil expenditure, the number of computers, and grade-span "
        "indicators. In this adjusted model, the coefficient on the "
        f"student–teacher ratio is {coef_full:.3f} (p-value = {pval_full:.3g}). "
        "A negative and statistically significant coefficient means that, within "
        "this dataset, lower student–teacher ratios are associated with higher "
        "academic performance even after accounting for these other factors. "
        f"Based on this evidence, I answer '{response}' to the question of "
        "whether lower student–teacher ratios are associated with higher "
        "academic performance."
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write required JSON output with no extra content
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

