import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and overall test score
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation
    pearson_corr = df["str"].corr(df["testscr"])

    # Simple OLS: testscr ~ str
    X_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple OLS with key demographic and resource controls
    controls = [
        "income",
        "calworks",
        "lunch",
        "english",
        "expenditure",
        "computer",
    ]
    X_controls = sm.add_constant(df[["str"] + controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()

    # Print key results for manual inspection
    print("N rows:", len(df))
    print("Pearson correlation between STR and testscr:", pearson_corr)
    print("\nSimple OLS: testscr ~ str")
    print("Coefficient on str:", model_simple.params["str"])
    print("Std err:", model_simple.bse["str"])
    print("t-stat:", model_simple.tvalues["str"])
    print("p-value:", model_simple.pvalues["str"])
    print("R-squared:", model_simple.rsquared)

    print("\nMultiple OLS with controls: testscr ~ str + controls")
    print("Coefficient on str:", model_controls.params["str"])
    print("Std err:", model_controls.bse["str"])
    print("t-stat:", model_controls.tvalues["str"])
    print("p-value:", model_controls.pvalues["str"])
    print("R-squared:", model_controls.rsquared)

    # Also check non-parametric monotonic association via Spearman
    spearman_corr = df["str"].corr(df["testscr"], method="spearman")
    print("\nSpearman correlation between STR and testscr:", spearman_corr)


if __name__ == "__main__":
    main()
