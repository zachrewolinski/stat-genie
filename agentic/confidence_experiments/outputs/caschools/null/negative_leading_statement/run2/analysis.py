import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: students per teacher (lower is better).
    df["stratio"] = df["students"] / df["teachers"]

    # Overall academic performance: average of reading and math scores.
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in key fields (should be none, but be safe).
    df = df.dropna(subset=["stratio", "testscr"])

    # Basic summary.
    print("Number of districts:", len(df))
    print("Student–teacher ratio (mean, std):", df["stratio"].mean(), df["stratio"].std())
    print("Test score (mean, std):", df["testscr"].mean(), df["testscr"].std())

    # Pearson correlation between student–teacher ratio and test scores.
    r, pval = stats.pearsonr(df["stratio"], df["testscr"])
    print("\nPearson correlation (testscr vs stratio):")
    print("  r =", r)
    print("  p-value =", pval)

    # Simple linear regression: testscr ~ stratio.
    X = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model_simple = sm.OLS(y, X).fit()
    print("\nSimple regression: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression controlling for some key covariates.
    covariates = ["income", "calworks", "lunch", "english", "computer", "expenditure"]
    df_cov = df.dropna(subset=covariates)
    X_multi = sm.add_constant(df_cov[["stratio"] + covariates])
    y_multi = df_cov["testscr"]
    model_multi = sm.OLS(y_multi, X_multi).fit()
    print("\nMultiple regression: testscr ~ stratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

