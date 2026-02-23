import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest (if any)
    cols_basic = ["stratio", "avg_score"]
    cols_controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    df_basic = df[cols_basic].dropna()
    df_full = df[cols_basic + cols_controls].dropna()

    print("Number of observations (basic):", len(df_basic))
    print("Number of observations (with controls):", len(df_full))

    # Simple correlation
    corr = df_basic["stratio"].corr(df_basic["avg_score"])
    print("Pearson correlation between student-teacher ratio and avg_score:", corr)

    # Simple linear regression: avg_score ~ stratio
    X_simple = sm.add_constant(df_basic["stratio"])
    y_simple = df_basic["avg_score"]
    model_simple = sm.OLS(y_simple, X_simple).fit()
    print("\nSimple OLS regression: avg_score ~ stratio")
    print("Coefficient on stratio:", model_simple.params["stratio"])
    print("Std. error on stratio:", model_simple.bse["stratio"])
    print("t-stat on stratio:", model_simple.tvalues["stratio"])
    print("p-value on stratio:", model_simple.pvalues["stratio"])
    print("R-squared:", model_simple.rsquared)

    # Multiple regression with key controls
    X_full = df_full[["stratio"] + cols_controls]
    X_full = sm.add_constant(X_full)
    y_full = df_full["avg_score"]
    model_full = sm.OLS(y_full, X_full).fit()
    print("\nMultiple OLS regression with controls: avg_score ~ stratio + controls")
    print("Coefficient on stratio (full model):", model_full.params["stratio"])
    print("Std. error on stratio (full model):", model_full.bse["stratio"])
    print("t-stat on stratio (full model):", model_full.tvalues["stratio"])
    print("p-value on stratio (full model):", model_full.pvalues["stratio"])
    print("R-squared (full model):", model_full.rsquared)


if __name__ == "__main__":
    main()

