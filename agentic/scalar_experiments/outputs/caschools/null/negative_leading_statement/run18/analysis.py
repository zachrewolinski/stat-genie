import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio (students per teacher)
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: mean of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic descriptives
    print("Number of districts:", len(df))
    print("Student-teacher ratio (stratio) summary:")
    print(df["stratio"].describe(), end="\n\n")
    print("Test score (testscr) summary:")
    print(df["testscr"].describe(), end="\n\n")

    # Correlation between student-teacher ratio and test scores
    corr, pval = stats.pearsonr(df["stratio"], df["testscr"])
    print("Correlation between stratio and testscr:")
    print(f"  r = {corr:.3f}, p-value = {pval:.3g}", end="\n\n")

    # Simple linear regression: testscr ~ stratio
    y = df["testscr"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    print("Simple OLS: testscr ~ stratio")
    print(model_simple.summary(), end="\n\n")

    # Multiple regression with key controls for socioeconomic and language factors
    controls = ["calworks", "lunch", "english", "income", "computer", "expenditure"]
    X_controls = df[["stratio"] + controls].copy()
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(y, X_controls).fit()
    print("Multiple OLS: testscr ~ stratio + controls")
    print(model_controls.summary(), end="\n\n")

    # Report estimated effect of a 5-student reduction in stratio
    beta_stratio_simple = model_simple.params["stratio"]
    beta_stratio_controls = model_controls.params["stratio"]
    print(
        "Estimated change in testscr for a 5-student reduction in stratio "
        "(holding controls constant in the multiple model):"
    )
    print(f"  Simple model:    {(-5) * beta_stratio_simple:.3f} points")
    print(f"  With controls:   {(-5) * beta_stratio_controls:.3f} points")


if __name__ == "__main__":
    main()

