import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Student–teacher ratio (class size proxy)
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2

    # Basic correlations
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    corr_testscr = df["stratio"].corr(df["testscr"])

    print("=== Correlations with student–teacher ratio (stratio) ===")
    print(f"corr(stratio, read)    = {corr_read:.4f}")
    print(f"corr(stratio, math)    = {corr_math:.4f}")
    print(f"corr(stratio, testscr) = {corr_testscr:.4f}")
    print()

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("=== OLS: testscr ~ stratio ===")
    print(model_simple.summary())
    print()

    # Multivariate regression with key covariates to check robustness
    covariates = ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    df_cov = df[covariates].dropna()
    X_multi = sm.add_constant(df_cov.drop(columns=["stratio"]))
    # Put stratio first for easy reading
    X_multi.insert(1, "stratio", df_cov["stratio"].values)
    model_multi = sm.OLS(df.loc[df_cov.index, "testscr"], X_multi).fit()
    print("=== OLS: testscr ~ stratio + controls ===")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

