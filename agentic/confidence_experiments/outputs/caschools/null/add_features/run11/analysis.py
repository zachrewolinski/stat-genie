import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and an overall academic performance measure.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in key variables, if present.
    key_cols = ["stratio", "testscr", "income", "calworks", "lunch", "english", "expenditure"]
    df_model = df.dropna(subset=key_cols).copy()

    # Simple correlation between student-teacher ratio and test scores.
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple linear regression: testscr ~ stratio.
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression controlling for key socioeconomic and resource variables.
    controls = ["income", "calworks", "lunch", "english", "expenditure"]
    X_multi = sm.add_constant(df_model[["stratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()

    # Print key results for inspection.
    print("Number of observations:", len(df_model))
    print("\nCorrelation between student-teacher ratio and test scores:")
    print(f"  r = {r:.3f}, p-value = {p_corr:.3g}")

    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    print("\nSimple OLS: testscr ~ stratio")
    print(f"  Coefficient on stratio: {coef_simple:.3f}")
    print(f"  p-value: {p_simple:.3g}")
    print(f"  R-squared: {model_simple.rsquared:.3f}")

    coef_multi = model_multi.params["stratio"]
    p_multi = model_multi.pvalues["stratio"]
    print("\nMultiple OLS: testscr ~ stratio + controls")
    print(f"  Coefficient on stratio (with controls): {coef_multi:.3f}")
    print(f"  p-value: {p_multi:.3g}")
    print(f"  R-squared: {model_multi.rsquared:.3f}")

    # Also print standard deviation of key variables for context.
    print("\nStandard deviations:")
    print(df_model[["stratio", "testscr"] + controls].std())


if __name__ == "__main__":
    main()

