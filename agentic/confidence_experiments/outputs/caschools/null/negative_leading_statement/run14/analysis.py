import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return {
        "coef_const": float(model.params["const"]),
        "coef_stratio": float(model.params["stratio"]),
        "pvalue_stratio": float(model.pvalues["stratio"]),
        "r2": float(model.rsquared),
    }


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["stratio", "testscr"]).copy()

    corr = df["stratio"].corr(df["testscr"])

    simple_full = fit_ols(df["testscr"], df[["stratio"]])

    covariates = [
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "students",
    ]
    available_covariates = [c for c in covariates if c in df.columns]
    multi_full = fit_ols(df["testscr"], df[["stratio"] + available_covariates])

    quantiles = df["stratio"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    # Trim to inter-quartile range of stratio for robustness
    iqr_mask = (df["stratio"] >= quantiles.loc[0.25]) & (df["stratio"] <= quantiles.loc[0.75])
    df_iqr = df.loc[iqr_mask].copy()

    corr_iqr = df_iqr["stratio"].corr(df_iqr["testscr"])
    simple_iqr = fit_ols(df_iqr["testscr"], df_iqr[["stratio"]])
    multi_iqr = fit_ols(df_iqr["testscr"], df_iqr[["stratio"] + available_covariates])

    results = {
        "n_obs_full": int(df.shape[0]),
        "n_obs_iqr": int(df_iqr.shape[0]),
        "stratio": {
            "mean": float(df["stratio"].mean()),
            "std": float(df["stratio"].std()),
            "min": float(df["stratio"].min()),
            "max": float(df["stratio"].max()),
            "quantiles": {str(k): float(v) for k, v in quantiles.items()},
        },
        "testscr": {
            "mean": float(df["testscr"].mean()),
            "std": float(df["testscr"].std()),
            "min": float(df["testscr"].min()),
            "max": float(df["testscr"].max()),
        },
        "pearson_corr_full": float(corr),
        "pearson_corr_iqr": float(corr_iqr),
        "simple_ols_full": simple_full,
        "multiple_ols_full": {**multi_full, "covariates": available_covariates},
        "simple_ols_iqr": simple_iqr,
        "multiple_ols_iqr": {**multi_iqr, "covariates": available_covariates},
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
