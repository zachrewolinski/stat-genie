import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: lower values mean smaller classes.
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = df[["read", "math"]].mean(axis=1)

    # Simple correlations
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    corr_avg = df["stratio"].corr(df["avgscore"])

    print("Simple correlations (stratio vs scores):")
    print(f"  read: {corr_read:.4f}")
    print(f"  math: {corr_math:.4f}")
    print(f"  avg : {corr_avg:.4f}")

    # Simple regression: avgscore ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avgscore"], X_simple).fit()

    print("\nSimple regression: avgscore ~ stratio")
    print(model_simple.summary().as_text())

    # Multiple regression with key controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer", "students"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(df["avgscore"], X_controls).fit()

    print("\nMultiple regression with controls: avgscore ~ stratio + controls")
    print(model_controls.summary().as_text())


if __name__ == "__main__":
    main()

