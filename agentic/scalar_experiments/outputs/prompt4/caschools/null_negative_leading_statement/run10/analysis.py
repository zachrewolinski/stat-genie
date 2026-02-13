import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlations between student-teacher ratio and achievement
    corr_testscr = df["stratio"].corr(df["testscr"])
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])

    print("Descriptive correlations (stratio vs outcomes):")
    print(f"  testscr: {corr_testscr:.4f}")
    print(f"  read   : {corr_read:.4f}")
    print(f"  math   : {corr_math:.4f}")
    print()

    # Simple bivariate regression: testscr ~ stratio
    y = df["testscr"]
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(y, X1).fit()

    print("Model 1: testscr ~ stratio")
    print(f"  Coefficient on stratio: {model1.params['stratio']:.4f}")
    print(f"  Std. error           : {model1.bse['stratio']:.4f}")
    print(f"  t-statistic          : {model1.tvalues['stratio']:.4f}")
    print(f"  p-value              : {model1.pvalues['stratio']:.4g}")
    print(f"  R-squared            : {model1.rsquared:.4f}")
    print()

    # Multiple regression with demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X2 = df[["stratio"] + controls].copy()
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()

    print("Model 2: testscr ~ stratio + controls")
    print(f"  Coefficient on stratio: {model2.params['stratio']:.4f}")
    print(f"  Std. error           : {model2.bse['stratio']:.4f}")
    print(f"  t-statistic          : {model2.tvalues['stratio']:.4f}")
    print(f"  p-value              : {model2.pvalues['stratio']:.4g}")
    print(f"  R-squared            : {model2.rsquared:.4f}")


if __name__ == "__main__":
    main()

