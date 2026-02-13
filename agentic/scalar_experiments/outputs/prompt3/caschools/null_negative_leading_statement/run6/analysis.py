import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    if "testscr" not in df.columns:
        df["testscr"] = (df["read"] + df["math"]) / 2.0

    if "str" not in df.columns:
        df["str"] = df["students"] / df["teachers"]

    # Drop any missing values in the main variables
    df_model = df[["testscr", "str", "income", "english", "lunch", "calworks"]].dropna()

    # Descriptive statistics
    corr_pearson, corr_p = stats.pearsonr(df_model["str"], df_model["testscr"])

    # Simple bivariate regression: testscr ~ str
    y = df_model["testscr"]
    X_simple = sm.add_constant(df_model["str"])
    model_simple = sm.OLS(y, X_simple).fit(cov_type="HC1")

    # Multiple regression controlling for observed covariates
    X_controls = df_model[["str", "income", "english", "lunch", "calworks"]]
    X_controls = sm.add_constant(X_controls)
    model_multi = sm.OLS(y, X_controls).fit(cov_type="HC1")

    # Summarize key results for manual inspection
    summary = {
        "n_obs": int(df_model.shape[0]),
        "corr_str_testscr": float(corr_pearson),
        "corr_p_value": float(corr_p),
        "simple_coef_str": float(model_simple.params["str"]),
        "simple_p_value_str": float(model_simple.pvalues["str"]),
        "simple_r2": float(model_simple.rsquared),
        "multi_coef_str": float(model_multi.params["str"]),
        "multi_p_value_str": float(model_multi.pvalues["str"]),
        "multi_r2": float(model_multi.rsquared),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

