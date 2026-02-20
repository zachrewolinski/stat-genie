import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Compute student-teacher ratio and combined test score
    df["stratio"] = df["students"] / df["teachers"]
    df["tests"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_correlation(df: pd.DataFrame, label: str = "full sample") -> None:
    r, p = stats.pearsonr(df["stratio"], df["tests"])
    print(f"Simple Pearson correlation between STR and tests ({label})")
    print(f"  r = {r:.3f}, p-value = {p:.4g}")


def simple_regression(df: pd.DataFrame, label: str = "full sample") -> None:
    X = sm.add_constant(df["stratio"])
    y = df["tests"]
    model = sm.OLS(y, X).fit()
    print(f"\nSimple OLS ({label}): tests ~ stratio")
    print(model.summary())


def multivariate_regression(df: pd.DataFrame, label: str = "full sample") -> None:
    covariates = ["stratio", "income", "english", "calworks", "lunch", "computer", "expenditure"]
    available = [c for c in covariates if c in df.columns]
    X = sm.add_constant(df[available])
    y = df["tests"]
    model = sm.OLS(y, X).fit()
    print(f"\nMultivariate OLS ({label}): tests ~ stratio + controls")
    print(model.summary())


def trimmed_analysis(df: pd.DataFrame) -> None:
    # Trim extreme outliers in student-teacher ratio (5th–95th percentile)
    q_low, q_high = df["stratio"].quantile([0.05, 0.95])
    trimmed = df[(df["stratio"] >= q_low) & (df["stratio"] <= q_high)].copy()
    print("\nTrimmed sample based on STR (5th–95th percentiles):")
    print(f"  STR range: {trimmed['stratio'].min():.2f} to {trimmed['stratio'].max():.2f}")
    print(f"  N districts in trimmed sample: {len(trimmed)}")

    simple_correlation(trimmed, label="trimmed STR")
    simple_regression(trimmed, label="trimmed STR")
    multivariate_regression(trimmed, label="trimmed STR")


def main() -> None:
    df = load_data("caschools.csv")
    print(f"Number of districts: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nSummary of key variables:")
    print(df[["stratio", "tests", "income", "english", "calworks", "lunch"]].describe())

    simple_correlation(df)
    simple_regression(df)
    multivariate_regression(df)

    trimmed_analysis(df)


if __name__ == "__main__":
    main()
