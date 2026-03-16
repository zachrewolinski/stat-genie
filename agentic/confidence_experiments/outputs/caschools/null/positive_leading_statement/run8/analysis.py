import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent

    info_path = base_path / "info.json"
    data_path = base_path / "caschools.csv"

    with info_path.open() as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Research question:")
    print(research_question)
    print()

    print("Basic description of key variables (str, testscr):")
    print(df[["str", "testscr"]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print()

    corr_testscr = df["str"].corr(df["testscr"])
    corr_read = df["str"].corr(df["read"])
    corr_math = df["str"].corr(df["math"])
    print(f"Pearson correlation between str and testscr: {corr_testscr:.4f}")
    print(f"Pearson correlation between str and reading score: {corr_read:.4f}")
    print(f"Pearson correlation between str and math score: {corr_math:.4f}")
    print()

    # Sensitivity: trim extreme student-teacher ratios to focus on typical range
    lower, upper = df["str"].quantile([0.05, 0.95])
    df_trim = df[(df["str"] >= lower) & (df["str"] <= upper)].copy()
    trim_corr = df_trim["str"].corr(df_trim["testscr"])
    print(
        "After trimming extreme 5% tails of str "
        f"(str in [{lower:.3f}, {upper:.3f}]):"
    )
    print(f"  Correlation(str, testscr) = {trim_corr:.4f}")
    print()

    # Simple bivariate regression
    X_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("OLS regression: testscr ~ str")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for observed confounders
    covariates = [
        "income",
        "english",
        "calworks",
        "lunch",
        "expenditure",
        "computer",
        "students",
    ]
    X_multi = sm.add_constant(df[["str"] + covariates].astype(float))
    model_multi = sm.OLS(df["testscr"], X_multi).fit()

    print("OLS regression with controls: testscr ~ str + demographics/resources")
    print(model_multi.summary())
    print()

    # Extract key statistics for str
    simple_coef = model_simple.params["str"]
    simple_p = model_simple.pvalues["str"]

    multi_coef = model_multi.params["str"]
    multi_p = model_multi.pvalues["str"]

    print("Key statistics for student-teacher ratio (str):")
    print(
        f"Simple model: coef={simple_coef:.4f}, p-value={simple_p:.4g}, "
        f"per +1 student per teacher change."
    )
    print(
        f"With controls: coef={multi_coef:.4f}, p-value={multi_p:.4g}, "
        f"per +1 student per teacher change."
    )


if __name__ == "__main__":
    main()
