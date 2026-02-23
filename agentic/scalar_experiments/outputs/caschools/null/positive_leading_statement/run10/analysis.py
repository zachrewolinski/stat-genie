import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: number of students per teacher
    df["str"] = df["students"] / df["teachers"]
    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2

    print("Basic description of key variables:")
    print(df[["str", "testscr", "read", "math"]].describe())
    print()

    corr = df["str"].corr(df["testscr"])
    print(f"Correlation between student-teacher ratio and test score: {corr:.4f}")
    print()

    # Simple bivariate regression: testscr ~ str
    X1 = sm.add_constant(df["str"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    print("Model 1: testscr ~ str (bivariate)")
    print(f"  Coefficient on str: {model1.params['str']:.4f}")
    print(f"  Std. error:        {model1.bse['str']:.4f}")
    print(f"  t-statistic:       {model1.tvalues['str']:.4f}")
    print(f"  p-value:           {model1.pvalues['str']:.6f}")
    print(f"  R-squared:         {model1.rsquared:.4f}")
    print()

    # Multiple regression with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X2 = df[["str"] + controls]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df["testscr"], X2).fit()
    print("Model 2: testscr ~ str + controls")
    print(f"  Coefficient on str: {model2.params['str']:.4f}")
    print(f"  Std. error:        {model2.bse['str']:.4f}")
    print(f"  t-statistic:       {model2.tvalues['str']:.4f}")
    print(f"  p-value:           {model2.pvalues['str']:.6f}")
    print(f"  R-squared:         {model2.rsquared:.4f}")
    print()

    print("Full Model 2 summary:")
    print(model2.summary())

    # Robustness check: trim extreme student-teacher ratios (likely data quirks)
    trimmed = df[df["str"] <= 40].copy()
    print()
    print("Robustness check (trimmed sample, str <= 40):")
    print(f"  Number of districts: {len(trimmed)}")
    corr_trim = trimmed["str"].corr(trimmed["testscr"])
    print(f"  Correlation (trimmed): {corr_trim:.4f}")

    X1_trim = sm.add_constant(trimmed["str"])
    model1_trim = sm.OLS(trimmed["testscr"], X1_trim).fit()
    print("  Trimmed Model: testscr ~ str")
    print(f"    Coefficient on str: {model1_trim.params['str']:.4f}")
    print(f"    Std. error:        {model1_trim.bse['str']:.4f}")
    print(f"    t-statistic:       {model1_trim.tvalues['str']:.4f}")
    print(f"    p-value:           {model1_trim.pvalues['str']:.6f}")
    print(f"    R-squared:         {model1_trim.rsquared:.4f}")


if __name__ == "__main__":
    main()
