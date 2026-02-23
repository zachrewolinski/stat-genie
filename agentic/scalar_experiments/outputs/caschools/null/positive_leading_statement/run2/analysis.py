import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio (more students per teacher = larger value)
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    print("Number of districts:", len(df))
    print()

    print("Student-teacher ratio summary (stratio = students/teachers)")
    print(df["stratio"].describe())
    print()

    print("Average score summary")
    print(df["avg_score"].describe())
    print()

    # Simple Pearson correlations
    for outcome in ["avg_score", "read", "math"]:
        r, p = pearsonr(df["stratio"], df[outcome])
        print(f"Pearson correlation stratio vs {outcome}: r={r:.3f}, p={p:.3g}")
    print()

    # Simple OLS: outcome ~ stratio
    def run_simple_ols(outcome: str) -> None:
        y = df[outcome]
        X = sm.add_constant(df["stratio"])
        model = sm.OLS(y, X).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared
        print(
            f"OLS {outcome} ~ stratio "
            f"(coef for stratio={coef:.3f}, p={pval:.3g}, R^2={r2:.3f})"
        )

    for outcome in ["avg_score", "read", "math"]:
        run_simple_ols(outcome)
    print()

    # Multiple regression controlling for key demographics and resources
    controls = ["calworks", "lunch", "income", "english", "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]

    def run_multiple_ols(outcome: str) -> None:
        y = df[outcome]
        X = df[["stratio"] + available_controls]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared
        print(
            f"Multiple OLS {outcome} ~ stratio + controls "
            f"(coef for stratio={coef:.3f}, p={pval:.3g}, R^2={r2:.3f})"
        )

    for outcome in ["avg_score", "read", "math"]:
        run_multiple_ols(outcome)

    print()

    # Robustness check: trim extreme student-teacher ratios (1st–99th percentiles)
    lower, upper = df["stratio"].quantile([0.01, 0.99])
    trimmed = df[(df["stratio"] >= lower) & (df["stratio"] <= upper)].copy()
    print(
        f"Trimmed sample size (1st–99th percentile of stratio): {len(trimmed)} "
        f"(original {len(df)})"
    )
    print(f"Trimmed stratio range: {trimmed['stratio'].min():.2f}–{trimmed['stratio'].max():.2f}")

    for outcome in ["avg_score", "read", "math"]:
        r, p = pearsonr(trimmed["stratio"], trimmed[outcome])
        print(
            f"Trimmed Pearson correlation stratio vs {outcome}: r={r:.3f}, p={p:.3g}"
        )



if __name__ == "__main__":
    main()
