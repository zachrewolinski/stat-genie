import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main():
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing data in key variables if any
    key_cols = ["stratio", "testscr", "income", "calworks", "lunch", "english", "expenditure"]
    df_model = df.dropna(subset=key_cols)

    # Simple Pearson correlations on full data
    r_simple, p_simple = stats.pearsonr(df_model["stratio"], df_model["testscr"])
    r_read, p_read = stats.pearsonr(df_model["stratio"], df_model["read"])
    r_math, p_math = stats.pearsonr(df_model["stratio"], df_model["math"])

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    coef_stratio_simple = model_simple.params["stratio"]
    p_stratio_simple = model_simple.pvalues["stratio"]

    # Multiple regression: testscr ~ stratio + controls
    controls = ["income", "calworks", "lunch", "english", "expenditure"]
    X_multi = sm.add_constant(df_model[["stratio"] + controls])
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()
    coef_stratio_multi = model_multi.params["stratio"]
    p_stratio_multi = model_multi.pvalues["stratio"]

    print("Number of observations (full model):", len(df_model))
    print("Mean student–teacher ratio (full):", df_model["stratio"].mean())
    print("Mean test score (full):", df_model["testscr"].mean())
    print()
    print("Pearson correlation between stratio and testscr:")
    print(f"  r = {r_simple:.3f}, p = {p_simple:.3g}")
    print("Pearson correlation with reading score:")
    print(f"  r = {r_read:.3f}, p = {p_read:.3g}")
    print("Pearson correlation with math score:")
    print(f"  r = {r_math:.3f}, p = {p_math:.3g}")
    print()
    print("Simple OLS: testscr ~ stratio")
    print(f"  coef(stratio) = {coef_stratio_simple:.3f}, p = {p_stratio_simple:.3g}")
    print(f"  R^2 = {model_simple.rsquared:.3f}")
    print()
    print("Multiple OLS: testscr ~ stratio + controls")
    print(f"  coef(stratio) = {coef_stratio_multi:.3f}, p = {p_stratio_multi:.3g}")
    print(f"  R^2 = {model_multi.rsquared:.3f}")

    # Robustness check: trim extreme ratios (5th–95th percentiles)
    q5, q95 = df_model["stratio"].quantile([0.05, 0.95])
    df_trim = df_model[(df_model["stratio"] >= q5) & (df_model["stratio"] <= q95)]

    r_trim, p_trim = stats.pearsonr(df_trim["stratio"], df_trim["testscr"])

    X_trim = sm.add_constant(df_trim["stratio"])
    model_trim_simple = sm.OLS(df_trim["testscr"], X_trim).fit()

    X_trim_multi = sm.add_constant(df_trim[["stratio"] + controls])
    model_trim_multi = sm.OLS(df_trim["testscr"], X_trim_multi).fit()

    print()
    print("Trimmed sample (5th–95th percentiles of stratio):")
    print("  Number of observations:", len(df_trim))
    print("  Mean student–teacher ratio (trimmed):", df_trim["stratio"].mean())
    print("  Pearson r (trimmed) =", f"{r_trim:.3f}, p = {p_trim:.3g}")
    print("  Simple OLS coef(stratio) =", f"{model_trim_simple.params['stratio']:.3f}, p = {model_trim_simple.pvalues['stratio']:.3g}")
    print("  Multiple OLS coef(stratio) =", f"{model_trim_multi.params['stratio']:.3f}, p = {model_trim_multi.pvalues['stratio']:.3g}")


if __name__ == "__main__":
    main()
