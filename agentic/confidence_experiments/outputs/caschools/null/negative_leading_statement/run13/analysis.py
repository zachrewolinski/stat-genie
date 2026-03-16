import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Student–teacher ratio: students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: mean of reading and math
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_regression(df: pd.DataFrame):
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()
    return model


def multiple_regression(df: pd.DataFrame):
    # Include key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X = df[["stratio"] + controls].copy()
    X = sm.add_constant(X)
    model = sm.OLS(df["testscr"], X).fit()
    return model


def summarize_model(label: str, model) -> dict:
    params = model.params
    pvalues = model.pvalues
    coef = float(params["stratio"])
    pval = float(pvalues["stratio"])
    r2 = float(model.rsquared)
    return {
        "model": label,
        "coef_stratio": coef,
        "pvalue_stratio": pval,
        "rsquared": r2,
    }


def main():
    data_path = Path("caschools.csv")
    df = load_data(data_path)

    print(f"Loaded {len(df)} districts")
    print(df[["stratio", "testscr"]].describe())

    corr = df["stratio"].corr(df["testscr"])
    print(f"\nCorrelation between student–teacher ratio and test score: {corr:.3f}")

    simple = simple_regression(df)
    multiple = multiple_regression(df)

    simple_summary = summarize_model("simple", simple)
    multiple_summary = summarize_model("multiple", multiple)

    print("\nSimple OLS: testscr ~ stratio")
    print(f"  Coefficient on stratio: {simple_summary['coef_stratio']:.3f}")
    print(f"  p-value: {simple_summary['pvalue_stratio']:.4g}")
    print(f"  R-squared: {simple_summary['rsquared']:.3f}")

    print("\nMultiple OLS: testscr ~ stratio + controls")
    print(f"  Coefficient on stratio: {multiple_summary['coef_stratio']:.3f}")
    print(f"  p-value: {multiple_summary['pvalue_stratio']:.4g}")
    print(f"  R-squared: {multiple_summary['rsquared']:.3f}")

    # Also dump a machine-readable summary for reference (not the final conclusion file)
    result = {
        "correlation_stratio_testscr": float(corr),
        "simple_model": simple_summary,
        "multiple_model": multiple_summary,
    }
    Path("analysis_summary.json").write_text(json.dumps(result, indent=2))
    print("\nSaved analysis summary to analysis_summary.json")


if __name__ == "__main__":
    main()

