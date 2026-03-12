import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("hurricane.csv")

    # Identify columns by description
    # From info.json: feature4 = masculinity-femininity index (higher = more feminine)
    # feature6 = binary gender indicator (0 male, 1 female)
    # feature8 = deaths
    # feature5 = min pressure at landfall
    # feature7 = category
    # feature13 = max wind speed
    # feature2 = year

    # Ensure numeric
    num_cols = [
        "feature4",
        "feature6",
        "feature8",
        "feature5",
        "feature7",
        "feature13",
        "feature2",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic group stats by binary gender
    group_stats = (
        df.groupby("feature6")["feature8"]
        .agg(["count", "mean", "median"])
        .rename(index={0: "male", 1: "female"})
    )

    # Correlations (Spearman due to skew)
    corr_spearman = df[["feature4", "feature8"]].corr(method="spearman").iloc[0, 1]

    # Regression: log1p(deaths) on femininity + severity controls
    df = df.dropna(subset=["feature4", "feature8", "feature5", "feature7", "feature13", "feature2"])
    df["log_deaths"] = np.log1p(df["feature8"])

    X = df[["feature4", "feature5", "feature7", "feature13", "feature2"]]
    X = sm.add_constant(X)
    y = df["log_deaths"]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Extract coefficient and p-value for femininity index
    coef = model.params["feature4"]
    pval = model.pvalues["feature4"]

    results = {
        "n": int(df.shape[0]),
        "group_stats": group_stats.to_dict(),
        "spearman_corr": float(corr_spearman),
        "coef_feminity_logdeaths": float(coef),
        "pvalue_feminity": float(pval),
        "model_r2": float(model.rsquared),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
