import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("panda_nuts.csv")
    # Basic cleaning
    df = df.copy()
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    # Fit model: efficiency as nuts per second
    model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()

    print("Rows:", len(df))
    print("Efficiency mean:", df["efficiency"].mean())
    print("Model summary:\n", model.summary())
    print("P-values:\n", model.pvalues)


if __name__ == "__main__":
    main()
