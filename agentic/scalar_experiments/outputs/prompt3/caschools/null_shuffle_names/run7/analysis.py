import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_data():
    """Load caschools data and construct key variables using info.json semantics."""
    base_dir = Path(__file__).parent
    df = pd.read_csv(base_dir / "caschools.csv")

    # Column semantics from info.json:
    # english   -> total enrollment (students)
    # students  -> number of teachers
    # district  -> average reading score
    # expenditure -> average math score
    # income    -> district average income
    # rownames  -> percent English learners
    # computer  -> percent qualifying for reduced-price lunch

    enrollment = df["english"]
    teachers = df["students"]
    read_score = df["district"]
    math_score = df["expenditure"]

    # Core constructed variables
    stratio = enrollment / teachers  # students per teacher
    testscr = (read_score + math_score) / 2.0

    income = df["income"]
    el_pct = df["rownames"]
    lunch_pct = df["computer"]

    data = pd.DataFrame(
        {
            "stratio": stratio,
            "read": read_score,
            "math": math_score,
            "testscr": testscr,
            "income": income,
            "el_pct": el_pct,
            "lunch_pct": lunch_pct,
        }
    ).dropna()

    return data


def bivariate_association(data: pd.DataFrame):
    """Compute simple correlations between student-teacher ratio and performance."""
    corr_testscr = data["stratio"].corr(data["testscr"])
    corr_read = data["stratio"].corr(data["read"])
    corr_math = data["stratio"].corr(data["math"])

    # p-values using Pearson correlation test
    r_ts, p_ts = stats.pearsonr(data["stratio"], data["testscr"])
    r_r, p_r = stats.pearsonr(data["stratio"], data["read"])
    r_m, p_m = stats.pearsonr(data["stratio"], data["math"])

    summary = {
        "corr_testscr": corr_testscr,
        "p_testscr": p_ts,
        "corr_read": corr_read,
        "p_read": p_r,
        "corr_math": corr_math,
        "p_math": p_m,
    }
    return summary


def regression_analysis(data: pd.DataFrame):
    """Run OLS regression of performance on student-teacher ratio with controls."""
    model_df = data[["testscr", "stratio", "income", "el_pct", "lunch_pct"]].dropna()
    y = model_df["testscr"]
    X = model_df[["stratio", "income", "el_pct", "lunch_pct"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    coef_stratio = model.params["stratio"]
    p_stratio = model.pvalues["stratio"]
    r_squared = model.rsquared

    return {
        "coef_stratio": coef_stratio,
        "p_stratio": p_stratio,
        "r_squared": r_squared,
    }


def group_comparison(data: pd.DataFrame):
    """
    Compare mean performance between low and high student-teacher ratio districts.

    Low ratio: bottom quartile of stratio.
    High ratio: top quartile of stratio.
    """
    q1 = data["stratio"].quantile(0.25)
    q3 = data["stratio"].quantile(0.75)

    low_group = data[data["stratio"] <= q1]["testscr"]
    high_group = data[data["stratio"] >= q3]["testscr"]

    mean_low = low_group.mean()
    mean_high = high_group.mean()
    diff = mean_low - mean_high

    # t-test (Welch) for difference in means
    t_stat, p_val = stats.ttest_ind(low_group, high_group, equal_var=False)

    return {
        "mean_low": mean_low,
        "mean_high": mean_high,
        "diff": diff,
        "t_stat": t_stat,
        "p_val": p_val,
    }


def main():
    data = load_data()

    biv = bivariate_association(data)
    reg = regression_analysis(data)
    grp = group_comparison(data)

    results = {
        "n_obs": int(len(data)),
        "bivariate": biv,
        "regression": reg,
        "group_comparison": grp,
    }

    # Save full numeric results for inspection
    out_path = Path(__file__).parent / "analysis_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Also print a concise textual summary for quick viewing
    print("Observations:", results["n_obs"])
    print("Correlation stratio ~ testscr:", biv["corr_testscr"], "p=", biv["p_testscr"])
    print("OLS coef on stratio (testscr):", reg["coef_stratio"], "p=", reg["p_stratio"])
    print("R-squared:", reg["r_squared"])
    print("Mean testscr (low ratio, Q1):", grp["mean_low"])
    print("Mean testscr (high ratio, Q3):", grp["mean_high"])
    print("Difference (low - high):", grp["diff"], "p=", grp["p_val"])


if __name__ == "__main__":
    main()

