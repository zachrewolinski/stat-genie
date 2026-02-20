import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlations
    corr_testscr = df["stratio"].corr(df["testscr"])
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])

    print("Correlation (stratio, testscr):", corr_testscr)
    print("Correlation (stratio, read):   ", corr_read)
    print("Correlation (stratio, math):   ", corr_math)
    print()

    # Simple linear regressions
    X_simple = sm.add_constant(df["stratio"])
    y_testscr = df["testscr"]
    y_read = df["read"]
    y_math = df["math"]

    model_testscr = sm.OLS(y_testscr, X_simple).fit()
    model_read = sm.OLS(y_read, X_simple).fit()
    model_math = sm.OLS(y_math, X_simple).fit()

    print("OLS: testscr ~ stratio")
    print(model_testscr.summary())
    print()

    print("OLS: read ~ stratio")
    print(model_read.summary())
    print()

    print("OLS: math ~ stratio")
    print(model_math.summary())
    print()

    # Multiple regression controlling for key demographics
    controls = ["income", "english", "lunch", "calworks"]
    X_multi = sm.add_constant(df[["stratio"] + controls])

    model_testscr_multi = sm.OLS(y_testscr, X_multi).fit()
    model_read_multi = sm.OLS(y_read, X_multi).fit()
    model_math_multi = sm.OLS(y_math, X_multi).fit()

    print("OLS (multiple): testscr ~ stratio + controls")
    print(model_testscr_multi.summary())
    print()

    print("OLS (multiple): read ~ stratio + controls")
    print(model_read_multi.summary())
    print()

    print("OLS (multiple): math ~ stratio + controls")
    print(model_math_multi.summary())
    print()

    # Compact summary of key coefficients and p-values
    def coef_info(label: str, model, var: str = "stratio") -> None:
        coef = model.params[var]
        pval = model.pvalues[var]
        print(f"{label}: coef={coef:.4f}, p={pval:.4f}")

    print("Key coefficient summaries for stratio")
    coef_info("testscr ~ stratio", model_testscr)
    coef_info("read ~ stratio", model_read)
    coef_info("math ~ stratio", model_math)
    coef_info("testscr ~ stratio + controls", model_testscr_multi)
    coef_info("read ~ stratio + controls", model_read_multi)
    coef_info("math ~ stratio + controls", model_math_multi)


if __name__ == "__main__":
    main()
