import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio (students per teacher). Lower values = smaller classes.
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores.
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic sanity checks
    print("N rows:", len(df))
    print("Student–teacher ratio summary:")
    print(df["stratio"].describe())
    print()

    print("Test score summary:")
    print(df["testscr"].describe())
    print()

    # Correlation between student–teacher ratio and test scores
    corr = df[["stratio", "testscr"]].corr().loc["stratio", "testscr"]
    print("Correlation between STR and testscr:", corr)
    print()

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("Simple OLS: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Repeat analysis after trimming implausibly large ratios to reduce the impact of outliers
    trimmed = df[df["stratio"] <= 35].copy()
    print("Trimmed sample (STR <= 35): N =", len(trimmed))
    print("Trimmed STR summary:")
    print(trimmed["stratio"].describe())
    print()

    corr_trim = trimmed[["stratio", "testscr"]].corr().loc["stratio", "testscr"]
    print("Correlation (trimmed) between STR and testscr:", corr_trim)
    print()

    X_simple_trim = sm.add_constant(trimmed["stratio"])
    model_simple_trim = sm.OLS(trimmed["testscr"], X_simple_trim).fit()
    print("Simple OLS on trimmed sample: testscr ~ stratio")
    print(model_simple_trim.summary())
    print()

    # Multivariate regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_controls = df[["stratio"] + controls].copy()
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df["testscr"], X_controls).fit()
    print("Multivariate OLS: testscr ~ stratio + controls")
    print(model_controls.summary())
    print()

    # Multivariate regression on trimmed sample
    X_controls_trim = trimmed[["stratio"] + controls].copy()
    X_controls_trim = sm.add_constant(X_controls_trim)
    model_controls_trim = sm.OLS(trimmed["testscr"], X_controls_trim).fit()
    print("Multivariate OLS on trimmed sample: testscr ~ stratio + controls")
    print(model_controls_trim.summary())


if __name__ == "__main__":
    main()
