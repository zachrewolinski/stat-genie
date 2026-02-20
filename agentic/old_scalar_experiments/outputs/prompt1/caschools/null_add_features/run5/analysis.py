import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Compute student–teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables
    df_clean = df[["stratio", "testscr", "calworks", "lunch", "income", "english"]].dropna()

    # Simple bivariate association
    corr = df_clean[["stratio", "testscr"]].corr().loc["stratio", "testscr"]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_clean["stratio"], df_clean["testscr"]
    )

    # Multivariable linear regression controlling for key demographics
    X = df_clean[["stratio", "calworks", "lunch", "income", "english"]]
    X = sm.add_constant(X)
    y = df_clean["testscr"]
    model = sm.OLS(y, X).fit()

    print("Number of districts used:", len(df_clean))
    print()
    print("Test score summary:")
    print(df_clean["testscr"].describe())
    print()
    print("Correlation (testscr vs student-teacher ratio):", corr)
    print("Simple regression slope (testscr on ratio):", slope)
    print("Simple regression p-value:", p_value)
    print()
    print("Correlation (testscr vs income):", df_clean[["testscr", "income"]].corr().iloc[0, 1])
    print("Correlation (testscr vs lunch):", df_clean[["testscr", "lunch"]].corr().iloc[0, 1])
    print("Correlation (testscr vs english):", df_clean[["testscr", "english"]].corr().iloc[0, 1])
    print()
    print("Multivariable OLS coefficient for ratio:", model.params["stratio"])
    print("Multivariable OLS p-value for ratio:", model.pvalues["stratio"])
    print("Multivariable OLS R-squared:", model.rsquared)


if __name__ == "__main__":
    main()
