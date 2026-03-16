import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Define student-teacher ratio and academic performance measures
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Basic descriptive statistics
    desc = df[["stratio", "avg_score", "read", "math"]].describe()

    # Simple bivariate regression: average score on student-teacher ratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()

    # Multivariate regression controlling for key demographics and resources
    controls = ["calworks", "lunch", "income", "english", "computer", "expenditure"]
    # Keep rows without missing values in the variables we use
    model_df = df[["avg_score", "stratio"] + controls].dropna()
    X_full = sm.add_constant(model_df[["stratio"] + controls])
    model_full = sm.OLS(model_df["avg_score"], X_full).fit()

    # Correlations
    corr_str_avg = df["stratio"].corr(df["avg_score"])
    corr_str_read = df["stratio"].corr(df["read"])
    corr_str_math = df["stratio"].corr(df["math"])

    # Print key results to stdout for inspection
    print("Descriptive statistics for key variables:")
    print(desc)
    print("\nCorrelation between student-teacher ratio and scores:")
    print(f"  avg_score vs stratio: {corr_str_avg:.3f}")
    print(f"  read vs stratio:      {corr_str_read:.3f}")
    print(f"  math vs stratio:      {corr_str_math:.3f}")

    print("\nSimple OLS: avg_score ~ stratio")
    print(model_simple.summary())

    print("\nMultivariate OLS: avg_score ~ stratio + controls")
    print(model_full.summary())


if __name__ == "__main__":
    main()

