import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest (should be rare)
    cols_simple = ["avg_score", "stratio"]
    cols_full = cols_simple + [
        "income",
        "english",
        "calworks",
        "lunch",
        "expenditure",
        "computer",
    ]
    df_simple = df[cols_simple].dropna()
    df_full = df[cols_full].dropna()

    print("Number of observations (simple model):", len(df_simple))
    print("Number of observations (full model):  ", len(df_full))

    # Simple bivariate regression: avg_score ~ stratio
    y_simple = df_simple["avg_score"]
    X_simple = sm.add_constant(df_simple["stratio"])
    model_simple = sm.OLS(y_simple, X_simple).fit()

    print("\n=== Simple OLS: avg_score ~ stratio ===")
    print(model_simple.summary())

    # Multivariable regression controlling for key covariates
    y_full = df_full["avg_score"]
    X_full = df_full[
        ["stratio", "income", "english", "calworks", "lunch", "expenditure", "computer"]
    ]
    X_full = sm.add_constant(X_full)
    model_full = sm.OLS(y_full, X_full).fit()

    print("\n=== Multivariable OLS: avg_score ~ stratio + controls ===")
    print(model_full.summary())

    # Print the key coefficient and p-value for the student-teacher ratio
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    coef_full = model_full.params["stratio"]
    pval_full = model_full.pvalues["stratio"]

    print("\nKey results for student-teacher ratio (stratio):")
    print(f"  Simple model: coef = {coef_simple:.3f}, p-value = {pval_simple:.4f}")
    print(f"  Full model:   coef = {coef_full:.3f}, p-value = {pval_full:.4f}")


if __name__ == "__main__":
    main()

