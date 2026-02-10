import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Core variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing values in key fields
    df_model = df[["stratio", "avgscore"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Correlation
    corr = df_model["stratio"].corr(df_model["avgscore"])

    # Simple linear regression: avgscore ~ stratio
    X = sm.add_constant(df_model["stratio"])
    model = sm.OLS(df_model["avgscore"], X).fit()

    print("Number of observations:", len(df_model))
    print("Correlation (stratio vs avgscore):", corr)
    print("OLS coefficient for stratio:", model.params["stratio"])
    print("OLS p-value for stratio:", model.pvalues["stratio"])
    print("R-squared:", model.rsquared)


if __name__ == "__main__":
    main()

