import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation between student-teacher ratio and test scores
    corr = df["stratio"].corr(df["testscr"])

    # Simple bivariate regression
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()

    # Multivariate regression with plausible confounders
    # Demographics and resources that may affect both class size and achievement
    formula_controls = (
        "testscr ~ stratio + income + english + lunch + calworks "
        "+ computer + expenditure + C(grades)"
    )
    model_controls = smf.ols(formula_controls, data=df).fit()

    # Collect key statistics for interpretation
    results = {
        "n_obs": int(df.shape[0]),
        "stratio_mean": float(df["stratio"].mean()),
        "testscr_mean": float(df["testscr"].mean()),
        "corr_stratio_testscr": float(corr),
        "simple_coef_stratio": float(model_simple.params["stratio"]),
        "simple_pvalue_stratio": float(model_simple.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "controls_coef_stratio": float(model_controls.params["stratio"]),
        "controls_pvalue_stratio": float(model_controls.pvalues["stratio"]),
        "controls_r2": float(model_controls.rsquared),
    }

    # Print a compact summary so we can inspect from the shell
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

