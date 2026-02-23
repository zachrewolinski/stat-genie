import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Keep observations with complete data for the models
    simple_vars = ["stratio", "testscr"]
    full_vars = simple_vars + ["calworks", "lunch", "income", "english"]

    df_simple = df[simple_vars].dropna()
    df_full = df[full_vars].dropna()

    # Correlation between student-teacher ratio and test scores
    corr = df_simple["stratio"].corr(df_simple["testscr"])

    # Simple linear regression: testscr ~ stratio
    x_simple = sm.add_constant(df_simple["stratio"])
    model_simple = sm.OLS(df_simple["testscr"], x_simple).fit()

    # Multiple regression with common demographic controls
    x_full = sm.add_constant(df_full[["stratio", "calworks", "lunch", "income", "english"]])
    model_full = sm.OLS(df_full["testscr"], x_full).fit()

    results = {
        "n_simple": int(df_simple.shape[0]),
        "n_full": int(df_full.shape[0]),
        "corr_stratio_testscr": float(corr),
        "coef_simple_stratio": float(model_simple.params["stratio"]),
        "p_simple_stratio": float(model_simple.pvalues["stratio"]),
        "coef_full_stratio": float(model_full.params["stratio"]),
        "p_full_stratio": float(model_full.pvalues["stratio"]),
        "r2_simple": float(model_simple.rsquared),
        "r2_full": float(model_full.rsquared),
    }

    # Save key numerical results for inspection
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

