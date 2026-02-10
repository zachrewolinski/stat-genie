import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Define key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Simple bivariate OLS: test scores on student-teacher ratio
    y = df["testscr"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple, missing="drop").fit()

    coef_simple = model_simple.params["stratio"]
    pvalue_simple = model_simple.pvalues["stratio"]
    r_squared_simple = model_simple.rsquared

    corr = df["stratio"].corr(df["testscr"])

    print("Bivariate association between student-teacher ratio and test scores")
    print(f"Coefficient (students per teacher): {coef_simple:.4f}")
    print(f"P-value: {pvalue_simple:.4g}")
    print(f"R-squared: {r_squared_simple:.4f}")
    print(f"Pearson correlation: {corr:.4f}")
    print()

    # Multivariate OLS with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    X_controls = df[["stratio"] + controls].copy()
    X_controls = sm.add_constant(X_controls)
    model_multi = sm.OLS(y, X_controls, missing="drop").fit()

    coef_multi = model_multi.params["stratio"]
    pvalue_multi = model_multi.pvalues["stratio"]

    print("Multivariate association controlling for demographics/resources")
    print(f"Coefficient (students per teacher): {coef_multi:.4f}")
    print(f"P-value: {pvalue_multi:.4g}")
    print(f"R-squared: {model_multi.rsquared:.4f}")


if __name__ == "__main__":
    main()
