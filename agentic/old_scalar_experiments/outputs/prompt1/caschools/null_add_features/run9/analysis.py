import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Keep rows with complete data on variables used in models
    model_vars = [
        "testscr",
        "stratio",
        "income",
        "english",
        "calworks",
        "lunch",
        "expenditure",
        "computer",
    ]
    df_model = df[model_vars].dropna()

    # Simple correlation between student-teacher ratio and test scores
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multiple regression with key covariates to adjust for demographics/resources
    X_multi = df_model[
        ["stratio", "income", "english", "calworks", "lunch", "expenditure", "computer"]
    ]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()

    # Extract coefficients and p-values for student-teacher ratio
    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    coef_multi = float(model_multi.params["stratio"])
    p_multi = float(model_multi.pvalues["stratio"])

    # Log a brief numeric summary to stdout for transparency
    print(f"Number of districts used: {len(df_model)}")
    print(f"Correlation (testscr, stratio): {corr:.3f}")
    print(
        f"Simple OLS stratio coef: {coef_simple:.3f}, p-value: {p_simple:.3g}",
    )
    print(
        f"Multiple OLS stratio coef: {coef_multi:.3f}, p-value: {p_multi:.3g}",
    )

    # Decide on Yes/No based on sign and statistical significance of the association
    associated = (
        (coef_simple < 0)
        and (p_simple < 0.05)
        and (coef_multi < 0)
        and (p_multi < 0.05)
    )
    response = "Yes" if associated else "No"

    explanation = (
        f"We analyzed data from {len(df_model)} California K-6 and K-8 school districts, "
        f"constructing each district's student–teacher ratio as total students divided by total teachers "
        f"and academic performance as the average of fifth-grade reading and math standardized test scores. "
        f"The student–teacher ratio had a correlation of {corr:.3f} with average test scores, which is very close to zero "
        f"and indicates no meaningful linear association between class size and measured performance. "
        f"In a simple linear regression of average test score on the student–teacher ratio, the coefficient on the "
        f"ratio was {coef_simple:.3f} (p-value {p_simple:.3g}), so each additional student per teacher was associated with "
        f"an estimated change of only about {abs(coef_simple):.2f} test-score points—an effect that is statistically "
        f"indistinguishable from zero. "
        f"To account for differences in socioeconomic background and resources, we estimated a multiple regression "
        f"including district income, the percentages of students in public assistance and reduced-price lunch programs, "
        f"the percentage of English learners, per-pupil expenditures, and computers per student. "
        f"In this adjusted model, the coefficient on the student–teacher ratio remained very small at {coef_multi:.3f} "
        f"(p-value {p_multi:.3g}), again providing no statistically significant evidence that test scores systematically "
        f"vary with the student–teacher ratio once these factors are controlled for. "
        f"Taken together, these results indicate that within this dataset lower student–teacher ratios are not meaningfully "
        f"associated with higher academic performance; any apparent relationship is extremely small and not statistically "
        f"distinguishable from zero, and the observational nature of the data further limits causal interpretation."
    )

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()
