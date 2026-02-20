import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic summary statistics
    corr = df["stratio"].corr(df["testscr"])

    # Simple bivariate regression: testscr ~ stratio
    y = df["testscr"]
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression with common controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]

    X_full = sm.add_constant(df[["stratio"] + available_controls])
    model_full = sm.OLS(y, X_full).fit()

    # Extract key results
    simple_coef = model_simple.params["stratio"]
    simple_p = model_simple.pvalues["stratio"]

    full_coef = model_full.params["stratio"]
    full_p = model_full.pvalues["stratio"]

    # Mean test scores by quartiles of the student-teacher ratio
    df["stratio_quartile"] = pd.qcut(
        df["stratio"], 4, labels=False, duplicates="drop"
    )
    quartile_summary = (
        df.groupby("stratio_quartile")[["stratio", "testscr"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    print("Number of districts:", len(df))
    print("Mean test score:", df["testscr"].mean())
    print("Mean student-teacher ratio:", df["stratio"].mean())
    print("Correlation(stratio, testscr):", corr)
    print()
    print("Simple regression testscr ~ stratio")
    print("  Coef (stratio):", simple_coef)
    print("  p-value (stratio):", simple_p)
    print("  R-squared:", model_simple.rsquared)
    print()
    print("Multiple regression with controls:", available_controls)
    print("  Coef (stratio):", full_coef)
    print("  p-value (stratio):", full_p)
    print("  R-squared:", model_full.rsquared)
    print()
    print("Mean scores by student-teacher ratio quartile:")
    print(quartile_summary)


if __name__ == "__main__":
    main()
