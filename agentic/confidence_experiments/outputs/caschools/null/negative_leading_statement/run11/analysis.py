import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Create key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Basic descriptive statistics
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    corr_avg = df["stratio"].corr(df["avgscore"])

    # Simple bivariate regression: avgscore ~ stratio
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["avgscore"], X_simple).fit()

    # Multiple regression with key covariates to check robustness
    covariates = [
        "stratio",
        "income",
        "calworks",
        "lunch",
        "english",
        "expenditure",
        "computer",
        "students",
    ]
    X_full = sm.add_constant(df[covariates])
    model_full = sm.OLS(df["avgscore"], X_full).fit()

    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    coef_full = model_full.params["stratio"]
    pval_full = model_full.pvalues["stratio"]

    # Effect size: difference in predicted score for a 5-student change
    delta_score_simple = coef_simple * 5.0
    delta_score_full = coef_full * 5.0

    summary_info = {
        "n_obs": int(df.shape[0]),
        "corr_read": float(corr_read),
        "corr_math": float(corr_math),
        "corr_avg": float(corr_avg),
        "coef_simple": float(coef_simple),
        "pval_simple": float(pval_simple),
        "r2_simple": float(model_simple.rsquared),
        "coef_full": float(coef_full),
        "pval_full": float(pval_full),
        "r2_full": float(model_full.rsquared),
        "delta_score_simple_per_5_students": float(delta_score_simple),
        "delta_score_full_per_5_students": float(delta_score_full),
    }

    print(json.dumps(summary_info, indent=2))


if __name__ == "__main__":
    main()

