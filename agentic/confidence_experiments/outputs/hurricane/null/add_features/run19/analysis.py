import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm


def prepare_data(df: pd.DataFrame):
    # Keep relevant columns and drop missing values
    cols = ["masfem", "alldeaths", "wind", "min", "category", "ndam15"]
    d = df[cols].copy()
    d = d.dropna()

    # Transform skewed variables
    d["log_deaths"] = np.log1p(d["alldeaths"])
    d["log_ndam15"] = np.log1p(d["ndam15"])
    return d


def ols_model(d: pd.DataFrame):
    X = d[["masfem", "wind", "min", "category", "log_ndam15"]]
    X = sm.add_constant(X)
    y = d["log_deaths"]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


def poisson_model(d: pd.DataFrame):
    X = d[["masfem", "wind", "min", "category", "log_ndam15"]]
    X = sm.add_constant(X)
    y = d["alldeaths"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC3")
    return model


def main():
    df = pd.read_csv("hurricane.csv")
    d = prepare_data(df)

    # Simple correlation between femininity and deaths
    corr = d[["masfem", "alldeaths"]].corr(method="spearman").iloc[0, 1]

    ols = ols_model(d)
    pois = poisson_model(d)

    result = {
        "n": int(d.shape[0]),
        "spearman_corr_masfem_deaths": float(corr),
        "ols_coef_masfem": float(ols.params["masfem"]),
        "ols_pvalue_masfem": float(ols.pvalues["masfem"]),
        "ols_r2": float(ols.rsquared),
        "poisson_coef_masfem": float(pois.params["masfem"]),
        "poisson_pvalue_masfem": float(pois.pvalues["masfem"]),
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
