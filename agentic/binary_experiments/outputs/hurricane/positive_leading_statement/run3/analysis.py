import pandas as pd
import numpy as np
import statsmodels.api as sm


def run():
    df = pd.read_csv("hurricane.csv")

    # Basic cleaning
    df = df.copy()
    df["log_deaths"] = np.log1p(df["alldeaths"])

    # Select controls capturing storm severity
    controls = ["wind", "min", "category", "year"]

    # Drop rows with missing values in model vars
    model_df = df[["log_deaths", "masfem"] + controls].dropna()

    X = sm.add_constant(model_df[["masfem"] + controls])
    y = model_df["log_deaths"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Also test binary gender indicator
    model_df2 = df[["log_deaths", "gender_mf"] + controls].dropna()
    X2 = sm.add_constant(model_df2[["gender_mf"] + controls])
    y2 = model_df2["log_deaths"]
    model2 = sm.OLS(y2, X2).fit(cov_type="HC3")

    # Correlation for context
    corr = df[["masfem", "alldeaths"]].corr().iloc[0, 1]

    print("N (masfem model):", int(model_df.shape[0]))
    print(model.summary().tables[1])
    print("\nN (gender_mf model):", int(model_df2.shape[0]))
    print(model2.summary().tables[1])
    print("\nCorrelation (masfem vs alldeaths):", corr)


if __name__ == "__main__":
    run()
