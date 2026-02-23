import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    # Student–teacher ratio (higher means larger classes, so lower is better)
    df["str"] = df["students"] / df["teachers"]
    # Academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Keep relevant columns and drop any missing values
    cols = [
        "testscr",
        "str",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]
    df_model = df[cols].dropna()

    # Simple bivariate relationship: testscr ~ str
    X_simple = sm.add_constant(df_model["str"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    X_multi = df_model[
        [
            "str",
            "income",
            "english",
            "lunch",
            "calworks",
            "computer",
            "expenditure",
        ]
    ]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()

    # Correlation between student–teacher ratio and test scores
    corr = float(df_model["testscr"].corr(df_model["str"]))

    # Extract key statistics
    coef_simple = float(model_simple.params["str"])
    p_simple = float(model_simple.pvalues["str"])
    r2_simple = float(model_simple.rsquared)

    coef_multi = float(model_multi.params["str"])
    p_multi = float(model_multi.pvalues["str"])
    r2_multi = float(model_multi.rsquared)

    negative_assoc = (coef_simple < 0) and (coef_multi < 0) and (corr < 0)
    strong_sig = (p_simple < 0.01) and (p_multi < 0.01)
    moderate_sig = (p_simple < 0.05) and (p_multi < 0.05)

    # Map evidence strength to a 0–100 Likert score
    if negative_assoc and strong_sig:
        # Consistent, strongly significant negative association
        response = 85
    elif negative_assoc and moderate_sig:
        # Consistent negative association with conventional significance
        response = 70
    elif negative_assoc and (p_simple < 0.1 or p_multi < 0.1):
        # Direction is as hypothesized but only marginally significant
        response = 55
    else:
        # Little or no evidence of the hypothesized relationship
        response = 30

    direction_word = "negative" if coef_multi < 0 else "positive"

    explanation = (
        "I examined whether a lower student–teacher ratio is associated with higher academic "
        "performance using data on 420 California K-6 and K-8 districts. I defined the student–teacher "
        "ratio as total students divided by total teachers, and academic performance as the average of "
        "fifth-grade reading and math Stanford 9 test scores. The simple correlation between the "
        f"student–teacher ratio and test scores was {corr:.3f}, indicating a {direction_word} association. "
        "In a simple linear regression of test scores on the student–teacher ratio, the coefficient on the "
        f"ratio was {coef_simple:.2f} (p = {p_simple:.3g}, R^2 = {r2_simple:.3f}), so a one-student increase "
        "in the average class size is associated with a change in test scores of that magnitude. In a "
        "multiple regression controlling for district income, percent English learners, program participation "
        "(CalWorks and reduced-price lunch), computers per student, and expenditures per pupil, the "
        f"coefficient on the student–teacher ratio remained {direction_word} at {coef_multi:.2f} "
        f"(p = {p_multi:.3g}, R^2 = {r2_multi:.3f}). "
    )

    if negative_assoc and (p_simple < 0.05 and p_multi < 0.05):
        explanation += (
            "Because the association is consistently negative and statistically significant even after "
            "adjusting for these covariates, the data provide strong evidence that lower student–teacher "
            "ratios are associated with higher academic performance in this dataset."
        )
    elif negative_assoc and (p_simple < 0.1 or p_multi < 0.1):
        explanation += (
            "The estimated association is in the expected (negative) direction and is only marginally "
            "statistically significant, so the evidence for a relationship is suggestive but not strong."
        )
    else:
        explanation += (
            "The estimated relationship is not consistently in the expected direction or lacks conventional "
            "statistical significance, so this dataset does not provide strong evidence that lower "
            "student–teacher ratios are associated with higher academic performance."
        )

    output = {"response": int(response), "explanation": explanation}

    Path("conclusion.txt").write_text(json.dumps(output))


if __name__ == "__main__":
    main()

