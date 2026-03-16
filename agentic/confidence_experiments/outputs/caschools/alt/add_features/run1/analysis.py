import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2

    # Drop any rows with missing values in variables we use
    vars_used = [
        "stratio",
        "testscr",
        "calworks",
        "lunch",
        "income",
        "english",
        "expenditure",
    ]
    df_model = df[vars_used].dropna()

    # Correlation between student–teacher ratio and test scores
    corr, corr_p = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple and multivariable linear regressions
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()
    model_controls = smf.ols(
        "testscr ~ stratio + calworks + lunch + income + english + expenditure",
        data=df_model,
    ).fit()

    print("Number of observations:", len(df_model))
    print("Correlation stratio vs testscr:", corr)
    print("Correlation p-value:", corr_p)

    print("\nSimple regression (testscr ~ stratio)")
    print("  coef(stratio):", model_simple.params["stratio"])
    print("  p-value:", model_simple.pvalues["stratio"])
    print("  R-squared:", model_simple.rsquared)

    print("\nRegression with controls")
    print("  coef(stratio):", model_controls.params["stratio"])
    print("  p-value:", model_controls.pvalues["stratio"])
    print("  R-squared:", model_controls.rsquared)

    print("\nstratio summary:")
    print(df_model["stratio"].describe())
    print("\ntestscr summary:")
    print(df_model["testscr"].describe())


if __name__ == "__main__":
    main()

