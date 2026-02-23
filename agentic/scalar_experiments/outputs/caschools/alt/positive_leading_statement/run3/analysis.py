import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: students per teacher (class size proxy)
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic summary
    print("N:", len(df))
    print("Student–teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTest score summary:")
    print(df["testscr"].describe())

    # Correlation
    corr = df["stratio"].corr(df["testscr"])
    print("\nCorrelation between student–teacher ratio and test score:", corr)

    # Simple linear regression
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with key observed covariates
    formula_controls = (
        "testscr ~ stratio + income + english + lunch + calworks "
        "+ expenditure + computer + students"
    )
    model_controls = smf.ols(formula_controls, data=df).fit()
    print("\nOLS with controls:")
    print(model_controls.summary())

    # Print key coefficients and p-values for convenience
    print("\nKey results:")
    print("Simple model - stratio coef, p-value:",
          model_simple.params["stratio"], model_simple.pvalues["stratio"])
    print("Controlled model - stratio coef, p-value:",
          model_controls.params["stratio"], model_controls.pvalues["stratio"])


if __name__ == "__main__":
    main()

