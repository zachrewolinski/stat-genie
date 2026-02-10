import pathlib

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = pathlib.Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    if {"read", "math"}.issubset(df.columns):
        df["testscr"] = (df["read"] + df["math"]) / 2.0
    else:
        raise ValueError("Expected 'read' and 'math' columns for academic performance.")

    ratio = df["student_teacher_ratio"]
    testscr = df["testscr"]

    # Correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(ratio, testscr)

    # Simple linear regression: testscr ~ student_teacher_ratio
    X = sm.add_constant(ratio)
    model = sm.OLS(testscr, X).fit()
    slope = model.params["student_teacher_ratio"]
    p_slope = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    # Standardized effect size (beta) for additional intuition
    slope_std = slope * (ratio.std(ddof=1) / testscr.std(ddof=1))

    print("Number of districts:", len(df))
    print("Mean student-teacher ratio:", ratio.mean())
    print("Std student-teacher ratio:", ratio.std(ddof=1))
    print("Mean test score:", testscr.mean())
    print("Std test score:", testscr.std(ddof=1))
    print()
    print("Pearson r (ratio, testscr):", r)
    print("p-value (correlation):", p_corr)
    print()
    print("OLS slope (testscr vs ratio):", slope)
    print("p-value (slope):", p_slope)
    print("R-squared:", r_squared)
    print("Standardized slope (beta):", slope_std)

    # Map evidence to a Likert-style scalar in [-100, 100]
    # Positive values mean "Yes, lower ratio is associated with higher performance".
    # Negative values mean the opposite.
    direction = -np.sign(slope)  # negative slope => positive evidence for the research claim
    abs_r = abs(r)

    if np.isnan(direction) or np.isnan(abs_r):
        scalar = 0
    else:
        if p_slope > 0.10:
            magnitude = 0.0
        elif p_slope > 0.05:
            magnitude = abs_r * 30.0
        elif p_slope > 0.01:
            magnitude = abs_r * 60.0
        else:
            magnitude = abs_r * 100.0

        scalar = int(np.clip(direction * magnitude, -100, 100))

    conclusion_path = pathlib.Path("conclusion.txt")
    conclusion_path.write_text(f"{scalar}\n", encoding="utf-8")
    print()
    print("Likert-style scalar conclusion (written to conclusion.txt):", scalar)


if __name__ == "__main__":
    main()

