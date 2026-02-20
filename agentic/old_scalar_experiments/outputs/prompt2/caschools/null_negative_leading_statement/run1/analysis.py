import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio (students per teacher)
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance measures: reading, math, and their average
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in key variables (if any)
    key_cols = ["stratio", "testscr", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    df_clean = df.dropna(subset=key_cols).copy()

    # Basic descriptive statistics: correlations with student-teacher ratio
    corr_stratio_testscr, p_testscr = stats.pearsonr(df_clean["stratio"], df_clean["testscr"])
    corr_stratio_read, p_read = stats.pearsonr(df_clean["stratio"], df_clean["read"])
    corr_stratio_math, p_math = stats.pearsonr(df_clean["stratio"], df_clean["math"])

    # Simple OLS regression: testscr ~ stratio
    X_simple = sm.add_constant(df_clean["stratio"])
    model_simple = sm.OLS(df_clean["testscr"], X_simple).fit()

    # Multiple regression with key covariates to account for confounding
    covariates = ["stratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_multi = sm.add_constant(df_clean[covariates])
    model_multi = sm.OLS(df_clean["testscr"], X_multi).fit()

    # Print a concise summary for inspection from the shell
    print("Number of observations (clean):", len(df_clean))
    print("\nCorrelation between student-teacher ratio (students per teacher)")
    print("and academic performance measures:")
    print(f"  Average score (testscr): r = {corr_stratio_testscr:.3f}, p = {p_testscr:.3g}")
    print(f"  Reading score (read):   r = {corr_stratio_read:.3f}, p = {p_read:.3g}")
    print(f"  Math score (math):      r = {corr_stratio_math:.3f}, p = {p_math:.3g}")

    print("\nSimple OLS regression: testscr ~ stratio")
    coef_stratio_simple = model_simple.params["stratio"]
    p_stratio_simple = model_simple.pvalues["stratio"]
    print(f"  Coefficient on stratio = {coef_stratio_simple:.3f}")
    print(f"  p-value for stratio = {p_stratio_simple:.3g}")
    print(f"  R-squared = {model_simple.rsquared:.3f}")

    print("\nMultiple OLS regression: testscr ~ stratio + controls")
    coef_stratio_multi = model_multi.params["stratio"]
    p_stratio_multi = model_multi.pvalues["stratio"]
    print(f"  Coefficient on stratio (with controls) = {coef_stratio_multi:.3f}")
    print(f"  p-value for stratio (with controls) = {p_stratio_multi:.3g}")
    print(f"  R-squared = {model_multi.rsquared:.3f}")


if __name__ == "__main__":
    main()
