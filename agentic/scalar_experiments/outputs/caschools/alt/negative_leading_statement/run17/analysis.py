import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables we use (should be none, but be safe)
    vars_basic = ["str", "testscr"]
    vars_controls = [
        "lunch",
        "english",
        "income",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_model = df[vars_basic + vars_controls].dropna()

    # Simple Pearson correlation
    corr = df_model["str"].corr(df_model["testscr"])

    # Simple OLS: testscr ~ str
    y = df_model["testscr"]
    X1 = sm.add_constant(df_model[["str"]])
    model1 = sm.OLS(y, X1).fit()
    coef_str_m1 = float(model1.params["str"])
    p_str_m1 = float(model1.pvalues["str"])
    r2_m1 = float(model1.rsquared)

    # Multiple OLS with controls
    X2 = sm.add_constant(df_model[["str"] + vars_controls])
    model2 = sm.OLS(y, X2).fit()
    coef_str_m2 = float(model2.params["str"])
    p_str_m2 = float(model2.pvalues["str"])
    r2_m2 = float(model2.rsquared)

    # OLS with possible non-linearity: add squared term
    df_model["str_sq"] = df_model["str"] ** 2
    X3 = sm.add_constant(df_model[["str", "str_sq"] + vars_controls])
    model3 = sm.OLS(y, X3).fit()
    coef_str_m3 = float(model3.params["str"])
    coef_strsq_m3 = float(model3.params["str_sq"])
    p_str_m3 = float(model3.pvalues["str"])
    p_strsq_m3 = float(model3.pvalues["str_sq"])
    r2_m3 = float(model3.rsquared)

    summary = {
        "n_obs": int(df_model.shape[0]),
        "corr_str_testscr": corr,
        "model1_simple": {
            "coef_str": coef_str_m1,
            "p_str": p_str_m1,
            "r2": r2_m1,
        },
        "model2_controls": {
            "coef_str": coef_str_m2,
            "p_str": p_str_m2,
            "r2": r2_m2,
        },
        "model3_nonlinear": {
            "coef_str": coef_str_m3,
            "coef_str_sq": coef_strsq_m3,
            "p_str": p_str_m3,
            "p_str_sq": p_strsq_m3,
            "r2": r2_m3,
        },
    }

    # Write machine-readable summary to inspect from the agent
    with open("analysis_output.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

