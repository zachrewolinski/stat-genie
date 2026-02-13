import pandas as pd
import numpy as np
import statsmodels.api as sm


def load_data(path: str = "caschools.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Student–teacher ratio: students per teacher
    df["str"] = df["students"] / df["teachers"]
    # Overall test score as average of reading and math
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_correlations(df: pd.DataFrame) -> None:
    corr = df["str"].corr(df["testscr"])
    print("Simple Pearson correlation between STR and testscr:")
    print(f"  corr(str, testscr) = {corr:.3f}")


def simple_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(df["str"])
    y = df["testscr"]
    model = sm.OLS(y, X).fit()
    print("\nOLS regression: testscr ~ str")
    print(model.summary())
    return model


def trimmed_analysis(df: pd.DataFrame, max_str: float = 40.0) -> sm.regression.linear_model.RegressionResultsWrapper:
    trimmed = df[df["str"] <= max_str].copy()
    print(f"\nTrimmed analysis for STR <= {max_str} (n={len(trimmed)}):")
    corr = trimmed["str"].corr(trimmed["testscr"])
    print("  Pearson correlation (trimmed):")
    print(f"    corr(str, testscr) = {corr:.3f}")

    X = sm.add_constant(trimmed["str"])
    y = trimmed["testscr"]
    model = sm.OLS(y, X).fit()
    print("\n  OLS regression on trimmed sample: testscr ~ str")
    print(model.summary())
    return model


def regression_with_controls(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    # Common socioeconomic and demographic controls
    controls = ["english", "lunch", "income", "calworks"]
    X = df[["str"] + controls]
    X = sm.add_constant(X)
    y = df["testscr"]
    model = sm.OLS(y, X).fit()
    print("\nOLS regression with controls: testscr ~ str + english + lunch + income + calworks")
    print(model.summary())
    return model


def main() -> None:
    df = load_data()

    print("Number of districts:", len(df))
    print("STR (students per teacher) summary:")
    print(df["str"].describe())
    print("\nTest score summary (average of reading and math):")
    print(df["testscr"].describe())

    simple_correlations(df)

    simple_model = simple_regression(df)
    trimmed_model = trimmed_analysis(df)
    controls_model = regression_with_controls(df)

    # Print key coefficients for quick reference
    print("\nKey coefficients:")
    print(f"  Simple model STR coefficient: {simple_model.params['str']:.3f}")
    print(f"  Simple model STR p-value: {simple_model.pvalues['str']:.4f}")
    print(f"  Trimmed model STR coefficient: {trimmed_model.params['str']:.3f}")
    print(f"  Trimmed model STR p-value: {trimmed_model.pvalues['str']:.4f}")
    print(f"  Controlled model STR coefficient: {controls_model.params['str']:.3f}")
    print(f"  Controlled model STR p-value: {controls_model.pvalues['str']:.4f}")


if __name__ == "__main__":
    main()
