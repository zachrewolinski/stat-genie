import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic description
    print("Student-teacher ratio (str) summary:")
    print(df["str"].describe())
    print("\nTest score (testscr) summary:")
    print(df["testscr"].describe())

    # Correlations
    corr, pval = stats.pearsonr(df["str"], df["testscr"])
    print("\nCorrelation between str and testscr:")
    print(f"  r = {corr:.4f}, p-value = {pval:.4g}")

    corr_read, p_read = stats.pearsonr(df["str"], df["read"])
    corr_math, p_math = stats.pearsonr(df["str"], df["math"])
    print("\nCorrelation between str and read/math:")
    print(f"  read: r = {corr_read:.4f}, p = {p_read:.4g}")
    print(f"  math: r = {corr_math:.4f}, p = {p_math:.4g}")

    # Simple linear regression: testscr on str
    X_simple = sm.add_constant(df["str"])
    y = df["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nSimple OLS: testscr ~ str")
    print(model_simple.summary())

    # Multiple regression with key controls
    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    X_controls = sm.add_constant(df[["str"] + controls])
    model_controls = sm.OLS(y, X_controls).fit()
    print("\nMultiple OLS: testscr ~ str + controls")
    print(model_controls.summary())

    # Quadratic specification to check non-linearity
    df["str_sq"] = df["str"] ** 2
    X_quad = sm.add_constant(df[["str", "str_sq"] + controls])
    model_quad = sm.OLS(y, X_quad).fit()
    print("\nQuadratic OLS: testscr ~ str + str^2 + controls")
    print(model_quad.summary())


if __name__ == "__main__":
    main()

