import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Reconstruct key variables based on metadata in info.json:
    # - english: total enrollment (number of students)
    # - students: number of teachers
    # - district: average reading score
    # - expenditure: average math score
    #
    # Student–teacher ratio (class size proxy) and academic performance:
    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing key fields, if any.
    df_clean = df[["stratio", "testscr", "income", "rownames"]].dropna()

    # Simple correlation between student–teacher ratio and test scores.
    r_simple, p_simple = stats.pearsonr(df_clean["stratio"], df_clean["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_clean["stratio"])
    model_simple = sm.OLS(df_clean["testscr"], X_simple).fit()
    beta_str_simple = model_simple.params["stratio"]
    p_beta_simple = model_simple.pvalues["stratio"]

    # Multiple regression with key demographic controls:
    # income: district average income (in $1,000)
    # rownames: percent of English learners
    X_multi = df_clean[["stratio", "income", "rownames"]]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_clean["testscr"], X_multi).fit()
    beta_str_multi = model_multi.params["stratio"]
    p_beta_multi = model_multi.pvalues["stratio"]

    summary = {
        "n_obs": int(df_clean.shape[0]),
        "stratio": {
            "mean": float(df_clean["stratio"].mean()),
            "std": float(df_clean["stratio"].std()),
            "min": float(df_clean["stratio"].min()),
            "max": float(df_clean["stratio"].max()),
        },
        "testscr": {
            "mean": float(df_clean["testscr"].mean()),
            "std": float(df_clean["testscr"].std()),
            "min": float(df_clean["testscr"].min()),
            "max": float(df_clean["testscr"].max()),
        },
        "correlation": {
            "r_simple": float(r_simple),
            "p_simple": float(p_simple),
        },
        "regression_simple": {
            "beta_stratio": float(beta_str_simple),
            "p_value": float(p_beta_simple),
            "r_squared": float(model_simple.rsquared),
        },
        "regression_multiple": {
            "beta_stratio": float(beta_str_multi),
            "p_value": float(p_beta_multi),
            "r_squared": float(model_multi.rsquared),
        },
    }

    # Print JSON summary to stdout so the agent can inspect results.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

