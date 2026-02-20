import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: number of students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["score"] = df[["read", "math"]].mean(axis=1)

    print("Basic description")
    print(df[["stratio", "score"]].describe())
    print()

    # Correlation between student-teacher ratio and performance
    pearson_corr = df["stratio"].corr(df["score"], method="pearson")
    spearman_corr = df["stratio"].corr(df["score"], method="spearman")
    print(f"Pearson correlation (stratio vs score): {pearson_corr:.4f}")
    print(f"Spearman correlation (stratio vs score): {spearman_corr:.4f}")
    print()

    # Simple linear regression: score ~ stratio
    X = sm.add_constant(df["stratio"])
    y = df["score"]
    model_simple = sm.OLS(y, X).fit()
    print("Simple OLS: score ~ stratio")
    print(model_simple.summary())
    print()

    # Multiple regression with key covariates to check robustness
    covariates = ["income", "english", "calworks", "lunch", "computer", "expenditure"]
    X_multi = sm.add_constant(df[["stratio"] + covariates])
    model_multi = sm.OLS(y, X_multi).fit()
    print("Multiple OLS: score ~ stratio + controls")
    print(model_multi.summary())

    print()
    print("Robustness checks with restricted stratio ranges")
    for upper in [30, 50]:
        subset = df[df["stratio"] <= upper]
        if subset.empty:
            continue
        pearson = subset["stratio"].corr(subset["score"])
        print(f"Upper bound {upper}: n={len(subset)}, Pearson corr={pearson:.4f}")


if __name__ == "__main__":
    main()
