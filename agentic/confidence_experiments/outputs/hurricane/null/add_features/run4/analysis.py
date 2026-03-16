import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("hurricane.csv")
    df = df.copy()
    df["log_deaths"] = np.log1p(df["alldeaths"])
    df["wind_z"] = (df["wind"] - df["wind"].mean()) / df["wind"].std(ddof=0)
    df["min_z"] = (df["min"] - df["min"].mean()) / df["min"].std(ddof=0)

    results = {}

    # Simple association
    m1 = smf.ols("log_deaths ~ masfem", data=df).fit(cov_type="HC3")
    results["m1"] = {
        "coef_masfem": float(m1.params.get("masfem")),
        "p_masfem": float(m1.pvalues.get("masfem")),
        "r2": float(m1.rsquared),
    }

    # Controls for storm intensity
    m2 = smf.ols("log_deaths ~ masfem + wind_z + min_z + category + year", data=df).fit(
        cov_type="HC3"
    )
    results["m2"] = {
        "coef_masfem": float(m2.params.get("masfem")),
        "p_masfem": float(m2.pvalues.get("masfem")),
        "r2": float(m2.rsquared),
    }

    # Binary female name indicator
    m3 = smf.ols("log_deaths ~ gender_mf + wind_z + min_z + category + year", data=df).fit(
        cov_type="HC3"
    )
    results["m3"] = {
        "coef_gender_mf": float(m3.params.get("gender_mf")),
        "p_gender_mf": float(m3.pvalues.get("gender_mf")),
        "r2": float(m3.rsquared),
    }

    # Interaction with severity (wind)
    m4 = smf.ols("log_deaths ~ masfem * wind_z + min_z + category + year", data=df).fit(
        cov_type="HC3"
    )
    results["m4"] = {
        "coef_masfem": float(m4.params.get("masfem")),
        "p_masfem": float(m4.pvalues.get("masfem")),
        "coef_interaction": float(m4.params.get("masfem:wind_z")),
        "p_interaction": float(m4.pvalues.get("masfem:wind_z")),
        "r2": float(m4.rsquared),
    }

    # Correlations
    results["corr"] = {
        "pearson_masfem_alldeaths": float(df[["masfem", "alldeaths"]].corr().iloc[0, 1]),
        "spearman_masfem_alldeaths": float(
            df[["masfem", "alldeaths"]].corr(method="spearman").iloc[0, 1]
        ),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
