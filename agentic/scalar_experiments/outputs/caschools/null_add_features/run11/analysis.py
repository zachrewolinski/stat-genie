import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Core variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously invalid rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["stratio", "avg_score"])

    # Simple correlation
    corr = df["stratio"].corr(df["avg_score"])

    # Bivariate regression avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    beta_stratio_simple = model_simple.params["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for key demographics
    controls = []
    for col in ["income", "lunch", "calworks", "english", "expenditure"]:
        if col in df.columns:
            controls.append(col)

    if controls:
        X_multi = sm.add_constant(df[["stratio"] + controls].dropna())
        y_multi = df.loc[X_multi.index, "avg_score"]
        model_multi = sm.OLS(y_multi, X_multi).fit()
        beta_stratio_multi = model_multi.params["stratio"]
        r2_multi = model_multi.rsquared
    else:
        beta_stratio_multi = np.nan
        r2_multi = np.nan

    print("N =", len(df))
    print("Correlation(stratio, avg_score) =", corr)
    print("Simple OLS beta_stratio:", beta_stratio_simple, "R^2:", r2_simple)
    print("Multiple OLS beta_stratio:", beta_stratio_multi, "R^2:", r2_multi)


if __name__ == "__main__":
    main()

