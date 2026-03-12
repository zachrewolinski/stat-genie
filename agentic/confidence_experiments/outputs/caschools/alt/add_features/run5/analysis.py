import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used in models (if any)
    base_cols = ["avg_score", "stratio"]
    controls = ["calworks", "lunch", "computer", "expenditure", "income", "english"]

    # Simple bivariate OLS: avg_score ~ stratio
    df_base = df[base_cols].dropna()
    X_base = sm.add_constant(df_base["stratio"])
    y_base = df_base["avg_score"]
    model_base = sm.OLS(y_base, X_base).fit()

    print("Bivariate OLS: avg_score ~ stratio")
    print(model_base.summary())
    print()

    # Multiple regression with key demographic and resource controls
    cols_full = base_cols + controls
    df_full = df[cols_full].dropna()
    X_full = sm.add_constant(df_full[["stratio"] + controls])
    y_full = df_full["avg_score"]
    model_full = sm.OLS(y_full, X_full).fit()

    print("Multivariate OLS with controls")
    print(model_full.summary())
    print()

    # Correlation between stratio and avg_score
    corr = df[["stratio", "avg_score"]].corr().iloc[0, 1]
    print(f"Correlation (stratio, avg_score): {corr:.4f}")


if __name__ == "__main__":
    main()

