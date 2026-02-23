import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio: larger value = more students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance measure: average of reading and math scores
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Basic descriptives
    print("N observations:", len(df))
    print("Student–teacher ratio (students per teacher)")
    print(df["stratio"].describe())
    print()

    print("Average test score (mean of reading and math)")
    print(df["avg_score"].describe())
    print()

    # Correlation between ratio and performance
    r, p_corr = stats.pearsonr(df["stratio"], df["avg_score"])
    print("Correlation between student–teacher ratio and avg_score:")
    print(f"  r = {r:.3f}, p = {p_corr:.4g}")
    print(
        "  (Negative r means that more students per teacher is associated "
        "with lower scores.)"
    )
    print()

    # Helper for OLS
    def run_ols(y, X, label: str):
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        print(f"=== OLS: {label} ===")
        print(model.summary())
        print()
        if "stratio" in model.params.index:
            coef = model.params["stratio"]
            pval = model.pvalues["stratio"]
            print(
                f"Coefficient on stratio: {coef:.3f} "
                f"(p = {pval:.4g}); "
                "interpretation is score change for one extra student per teacher."
            )
            print()
        return model

    y = df["avg_score"]

    # Bivariate regression
    run_ols(y, df[["stratio"]], "avg_score ~ stratio")

    # Multiple regression with key demographic and resource controls
    controls = ["english", "lunch", "calworks", "income", "expenditure"]
    run_ols(y, df[["stratio"] + controls], "avg_score ~ stratio + controls")


if __name__ == "__main__":
    main()

