import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic summaries
    print("Number of observations:", len(df))
    print()
    print("Student-teacher ratio (stratio) summary:")
    print(df["stratio"].describe())
    print()
    print("Test score (testscr) summary:")
    print(df["testscr"].describe())
    print()

    # Simple Pearson and Spearman correlations
    pearson = df["testscr"].corr(df["stratio"], method="pearson")
    spearman = df["testscr"].corr(df["stratio"], method="spearman")
    print(f"Pearson correlation between testscr and stratio: {pearson:.4f}")
    print(f"Spearman correlation between testscr and stratio: {spearman:.4f}")
    print()

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    print("Simple OLS regression: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for key demographics
    # These controls follow the common CASchools specification.
    controls = ["income", "english", "lunch", "calworks"]
    formula_controls = "testscr ~ stratio + " + " + ".join(controls)
    model_controls = smf.ols(formula_controls, data=df).fit()
    print("Multiple OLS regression with controls:")
    print(formula_controls)
    print(model_controls.summary())
    print()

    # Robustness check: drop extreme outliers in stratio (1st and 99th percentiles)
    low, high = df["stratio"].quantile([0.01, 0.99])
    df_trim = df[(df["stratio"] >= low) & (df["stratio"] <= high)]

    model_trim = smf.ols("testscr ~ stratio", data=df_trim).fit()
    print("Trimmed sample OLS (1st–99th percentile of stratio): testscr ~ stratio")
    print(model_trim.summary())


if __name__ == "__main__":
    main()
