import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables of interest
    cols = [
        "stratio",
        "testscr",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_model = df.dropna(subset=cols).copy()

    # Simple correlation
    r, pval_r = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    beta_str_simple = model_simple.params["stratio"]
    pval_str_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression with key covariates
    X_multi = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    ]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()
    beta_str_multi = model_multi.params["stratio"]
    pval_str_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    # Summarise results for inspection
    summary = {
        "n_obs": int(df_model.shape[0]),
        "stratio": {
            "mean": float(df_model["stratio"].mean()),
            "std": float(df_model["stratio"].std()),
            "min": float(df_model["stratio"].min()),
            "max": float(df_model["stratio"].max()),
        },
        "testscr": {
            "mean": float(df_model["testscr"].mean()),
            "std": float(df_model["testscr"].std()),
            "min": float(df_model["testscr"].min()),
            "max": float(df_model["testscr"].max()),
        },
        "correlation": {
            "r": float(r),
            "p_value": float(pval_r),
        },
        "simple_regression": {
            "beta_stratio": float(beta_str_simple),
            "p_value_stratio": float(pval_str_simple),
            "r_squared": float(r2_simple),
        },
        "multiple_regression": {
            "beta_stratio": float(beta_str_multi),
            "p_value_stratio": float(pval_str_multi),
            "r_squared": float(r2_multi),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

