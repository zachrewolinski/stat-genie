import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = 0.5 * (df["read"] + df["math"])

    sub = df[["testscr", "stratio", "income", "english", "lunch"]].dropna()

    # Restrict to a more plausible range of student-teacher ratios to reduce
    # the influence of extreme outliers.
    trimmed = sub[(sub["stratio"] >= 5) & (sub["stratio"] <= 40)]

    r_trim, p_corr_trim = stats.pearsonr(trimmed["stratio"], trimmed["testscr"])

    X1 = sm.add_constant(trimmed["stratio"])
    model1 = sm.OLS(trimmed["testscr"], X1).fit()

    X2 = sm.add_constant(trimmed[["stratio", "income", "english", "lunch"]])
    model2 = sm.OLS(trimmed["testscr"], X2).fit()

    results = {
        "n_obs_trimmed": int(len(trimmed)),
        "stratio_summary_trimmed": {
            "mean": float(trimmed["stratio"].mean()),
            "std": float(trimmed["stratio"].std()),
            "min": float(trimmed["stratio"].min()),
            "max": float(trimmed["stratio"].max()),
        },
        "correlation_trimmed": {
            "r": float(r_trim),
            "p_value": float(p_corr_trim),
        },
        "bivariate_regression_trimmed": {
            "coef_stratio": float(model1.params["stratio"]),
            "p_value_stratio": float(model1.pvalues["stratio"]),
            "r_squared": float(model1.rsquared),
        },
        "multivariate_regression_trimmed": {
            "coef_stratio": float(model2.params["stratio"]),
            "p_value_stratio": float(model2.pvalues["stratio"]),
            "r_squared": float(model2.rsquared),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

