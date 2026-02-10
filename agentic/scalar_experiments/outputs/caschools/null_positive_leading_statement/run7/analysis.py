import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent


def load_data():
    df = pd.read_csv(ROOT / "caschools.csv")
    # Student-teacher ratio: more intuitive if defined as students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Composite academic performance: average of reading and math scores
    df["perf"] = df[["read", "math"]].mean(axis=1)
    return df


def simple_association(df: pd.DataFrame):
    corr = df["stratio"].corr(df["perf"])
    # OLS without controls
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["perf"], X).fit()
    return {
        "corr": float(corr),
        "coef": float(model.params["stratio"]),
        "pvalue": float(model.pvalues["stratio"]),
        "r2": float(model.rsquared),
    }


def controlled_association(df: pd.DataFrame):
    # Include key socioeconomic and demographic controls
    controls = [
        "income",  # average district income
        "lunch",  # % reduced-price lunch (poverty proxy)
        "english",  # % English learners
        "calworks",  # public assistance
    ]
    cols = ["stratio"] + controls
    # Drop rows with any missing data in model variables
    sub = df[["perf"] + cols].dropna()
    X = sm.add_constant(sub[cols])
    model = sm.OLS(sub["perf"], X).fit()
    return {
        "coef": float(model.params["stratio"]),
        "pvalue": float(model.pvalues["stratio"]),
        "r2": float(model.rsquared),
    }


def map_to_scalar(simple_stats, controlled_stats) -> int:
    """Map statistical evidence to a -100..100 Likert-style scalar.

    Heuristic:
    - Start from 0 (neutral) and adjust based on:
      * sign consistency between simple and controlled models
      * p-values
      * effect sizes and R^2
    """

    s_coef = simple_stats["coef"]
    c_coef = controlled_stats["coef"]

    # We expect a negative coefficient on stratio: more students per teacher
    # (larger ratio) should be associated with lower performance.
    sign_consistent = (s_coef < 0) and (c_coef < 0)

    s_p = simple_stats["pvalue"]
    c_p = controlled_stats["pvalue"]

    # Base score
    score = 0

    if sign_consistent:
        # Strength from significance
        # Very strong evidence in both models
        if s_p < 0.001 and c_p < 0.001:
            score = 85
        # Strong but not extreme
        elif s_p < 0.01 and c_p < 0.01:
            score = 70
        # Moderate evidence
        elif s_p < 0.05 and c_p < 0.05:
            score = 55
        # Weak/inconclusive but aligned
        elif s_p < 0.1 or c_p < 0.1:
            score = 30
        else:
            score = 10
    else:
        # Mixed signs or evidence
        if (s_p < 0.05 or c_p < 0.05):
            score = 0
        else:
            score = -10

    # Clip to [-100, 100] and convert to int
    score = int(max(-100, min(100, round(score))))
    return score


def main():
    df = load_data()
    simple_stats = simple_association(df)
    controlled_stats = controlled_association(df)

    scalar = map_to_scalar(simple_stats, controlled_stats)

    # Persist a brief log for transparency (not required by instructions)
    results = {
        "simple": simple_stats,
        "controlled": controlled_stats,
        "scalar": scalar,
    }
    with open(ROOT / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write the required scalar-only conclusion file
    with open(ROOT / "conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()
