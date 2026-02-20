import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest (should be none but keep safe)
    cols = ["stratio", "testscr", "income", "english", "lunch", "calworks", "expenditure"]
    df_model = df[cols].dropna()

    # Simple correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    ols_simple = sm.OLS(y, X_simple).fit()
    beta_str_simple = ols_simple.params["stratio"]
    p_str_simple = ols_simple.pvalues["stratio"]

    # Multiple OLS with key socioeconomic controls
    X_controls = df_model[["stratio", "income", "english", "lunch", "calworks", "expenditure"]]
    X_controls = sm.add_constant(X_controls)
    ols_controls = sm.OLS(y, X_controls).fit()
    beta_str_controls = ols_controls.params["stratio"]
    p_str_controls = ols_controls.pvalues["stratio"]

    # Print a compact summary of key results
    print("N used:", len(df_model))
    print("Correlation stratio vs testscr: r = {:.3f}, p = {:.4g}".format(r, p_corr))
    print("Simple OLS: testscr ~ stratio")
    print("  beta_stratio = {:.3f}, p = {:.4g}".format(beta_str_simple, p_str_simple))
    print("Multiple OLS with controls (income, english, lunch, calworks, expenditure)")
    print("  beta_stratio = {:.3f}, p = {:.4g}".format(beta_str_controls, p_str_controls))


if __name__ == "__main__":
    main()

