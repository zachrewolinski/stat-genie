import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]  # student–teacher ratio
    df["testscr"] = (df["read"] + df["math"]) / 2.0  # overall test score

    # Basic correlations
    corr_str_testscr = df["str"].corr(df["testscr"])

    # Simple linear regression: testscr ~ str
    X_simple = sm.add_constant(df[["str"]])
    simple_model = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression with key demographic controls
    controls = ["english", "lunch", "calworks", "income"]
    X_controls = sm.add_constant(df[["str"] + controls])
    multi_model = sm.OLS(df["testscr"], X_controls).fit()

    # Print summary statistics relevant for interpretation
    print("Correlation between student–teacher ratio (str) and test score:")
    print(f"  corr(str, testscr) = {corr_str_testscr:.3f}")
    print()

    print("Simple regression: testscr ~ str")
    print(f"  coef(str) = {simple_model.params['str']:.3f}")
    print(f"  p-value(str) = {simple_model.pvalues['str']:.3g}")
    print(f"  R-squared = {simple_model.rsquared:.3f}")
    print()

    print("Multiple regression: testscr ~ str + controls")
    print(f"  coef(str) = {multi_model.params['str']:.3f}")
    print(f"  p-value(str) = {multi_model.pvalues['str']:.3g}")
    print(f"  R-squared = {multi_model.rsquared:.3f}")


if __name__ == "__main__":
    main()

