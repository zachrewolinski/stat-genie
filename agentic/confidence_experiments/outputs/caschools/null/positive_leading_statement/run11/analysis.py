import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously problematic rows (e.g., missing or non-finite values)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["str", "testscr", "income", "english", "lunch", "calworks", "expenditure"]
    )

    # Basic descriptive statistics
    desc = df[["str", "testscr"]].describe()

    # Simple bivariate relationship: Pearson correlation and OLS
    corr = df["str"].corr(df["testscr"])

    X_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression controlling for observed covariates
    covariates = ["str", "income", "english", "lunch", "calworks", "expenditure"]
    X_full = sm.add_constant(df[covariates])
    model_full = sm.OLS(df["testscr"], X_full).fit()

    # Print key results to stdout so the agent can inspect them
    print("Descriptive statistics for STR and test scores:")
    print(desc)
    print("\nPearson correlation between STR and testscr:")
    print(f"corr(str, testscr) = {corr:.4f}")

    print("\nSimple OLS: testscr ~ str")
    print(model_simple.summary())

    print("\nMultiple OLS: testscr ~ str + income + english + lunch + calworks + expenditure")
    print(model_full.summary())

    # Robustness check: trim extreme STR values (5th–95th percentile)
    q_low, q_high = df["str"].quantile([0.05, 0.95])
    trimmed = df[(df["str"] >= q_low) & (df["str"] <= q_high)].copy()

    print("\n\nRobustness check with trimmed STR (5th–95th percentile):")
    print(f"STR quantiles used for trimming: 5th={q_low:.2f}, 95th={q_high:.2f}")
    corr_trim = trimmed["str"].corr(trimmed["testscr"])
    print(f"corr(str, testscr) [trimmed] = {corr_trim:.4f}")

    X_trim_simple = sm.add_constant(trimmed["str"])
    model_trim_simple = sm.OLS(trimmed["testscr"], X_trim_simple).fit()
    print("\nTrimmed simple OLS: testscr ~ str")
    print(model_trim_simple.summary())

    X_trim_full = sm.add_constant(trimmed[covariates])
    model_trim_full = sm.OLS(trimmed["testscr"], X_trim_full).fit()
    print("\nTrimmed multiple OLS: testscr ~ str + income + english + lunch + calworks + expenditure")
    print(model_trim_full.summary())


if __name__ == "__main__":
    main()
