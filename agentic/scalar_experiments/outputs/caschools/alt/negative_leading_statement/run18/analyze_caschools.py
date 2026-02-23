import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest (there should be none, but be safe)
    vars_of_interest = [
        "stratio",
        "testscr",
        "read",
        "math",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
        "students",
    ]
    df_clean = df.dropna(subset=vars_of_interest).copy()

    # Simple correlations between student-teacher ratio and performance measures
    corr_testscr, p_testscr = stats.pearsonr(df_clean["stratio"], df_clean["testscr"])
    corr_read, p_read = stats.pearsonr(df_clean["stratio"], df_clean["read"])
    corr_math, p_math = stats.pearsonr(df_clean["stratio"], df_clean["math"])

    print("Bivariate correlations (stratio vs performance):")
    print(f"  testscr: r = {corr_testscr:.3f}, p = {p_testscr:.4g}")
    print(f"  read   : r = {corr_read:.3f}, p = {p_read:.4g}")
    print(f"  math   : r = {corr_math:.3f}, p = {p_math:.4g}")
    print()

    # Simple OLS: testscr on stratio
    X_simple = sm.add_constant(df_clean["stratio"])
    y = df_clean["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    print("Simple OLS: testscr ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for observed covariates
    controls = [
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
        "students",
    ]
    X_controls = sm.add_constant(df_clean[["stratio"] + controls])
    model_controls = sm.OLS(y, X_controls).fit()

    print("Multiple OLS with controls: testscr ~ stratio + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

