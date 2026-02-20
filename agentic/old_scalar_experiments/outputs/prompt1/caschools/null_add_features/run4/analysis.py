import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    base_vars = ["testscr", "stratio"]
    control_vars = ["income", "calworks", "lunch", "english", "computer", "expenditure"]
    vars_for_models = base_vars + control_vars
    df_model = df[vars_for_models].dropna()

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model[["stratio"]])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression with controls
    X_controls = sm.add_constant(df_model[["stratio"] + control_vars])
    model_controls = sm.OLS(y, X_controls).fit()
    coef_controls = float(model_controls.params["stratio"])
    pval_controls = float(model_controls.pvalues["stratio"])

    # Determine answer based on sign and significance of coefficient on stratio.
    # A negative coefficient implies that lower student-teacher ratios
    # (fewer students per teacher) are associated with higher performance.
    alpha = 0.05
    is_negative_and_significant = (coef_controls < 0.0) and (pval_controls < alpha)

    response = "Yes" if is_negative_and_significant else "No"

    explanation_parts = [
        "I examined the association between student-teacher ratio and academic performance "
        "in the California K-6 and K-8 districts dataset.",
        f"I constructed a student-teacher ratio variable as students divided by teachers "
        f"and an average test score combining reading and math (testscr).",
        f"A simple linear regression of testscr on the student-teacher ratio produced a "
        f"coefficient of {coef_simple:.3f} with p-value {pval_simple:.3g}.",
        "I also estimated a multiple regression including controls for district income, "
        "CalWorks percentage, reduced-price-lunch percentage, English-learner percentage, "
        "computers per student, and expenditure per student.",
        f"In this controlled model, the coefficient on the student-teacher ratio was "
        f"{coef_controls:.3f} with p-value {pval_controls:.3g}.",
    ]

    if is_negative_and_significant:
        explanation_parts.append(
            "The negative and statistically significant coefficient means that districts with "
            "lower student-teacher ratios (fewer students per teacher) tend to have higher "
            "average test scores, even after accounting for these covariates."
        )
    else:
        explanation_parts.append(
            "Because the coefficient on the student-teacher ratio is not both negative and "
            "statistically significant in the controlled model, the data do not provide clear "
            "evidence that lower student-teacher ratios are associated with higher academic "
            "performance once other factors are taken into account."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

