import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_support_scalar(beta: float, p_value: float, stratio: pd.Series, testscr: pd.Series) -> int:
    """Map regression results to a [-100, 100] scalar where positive supports 'Yes'."""
    if pd.isna(beta) or pd.isna(p_value):
        return 0

    std_ratio = float(stratio.std(ddof=1))
    std_test = float(testscr.std(ddof=1))

    if std_ratio <= 0 or std_test <= 0:
        d = 0.0
    else:
        # Standardized effect size: change in testscr (in SDs) for 1 SD change in ratio.
        d = abs(beta) * std_ratio / std_test

    # Normalize effect size: d ≈ 0.5 (medium) or larger is treated as "strong".
    d_norm = min(d / 0.5, 1.0)

    # Weight by statistical significance.
    if p_value < 0.001:
        sig_weight = 1.0
    elif p_value < 0.01:
        sig_weight = 0.8
    elif p_value < 0.05:
        sig_weight = 0.6
    elif p_value < 0.1:
        sig_weight = 0.3
    else:
        sig_weight = 0.1

    # Negative coefficient means lower ratio is associated with higher scores,
    # which supports a "Yes" answer.
    effect_sign = 1.0 if beta < 0 else -1.0

    support_continuous = d_norm * sig_weight * effect_sign
    scalar = int(round(100 * support_continuous))
    scalar = max(-100, min(100, scalar))
    return scalar


def main() -> None:
    # Load metadata (not strictly required for the analysis logic, but keeps us aligned with instructions).
    info_path = Path("info.json")
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
        print("Research questions:", info.get("research_questions"))

    # Load dataset.
    df = pd.read_csv("caschools.csv")

    # Construct key variables.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used below.
    base_cols = ["testscr", "stratio"]
    control_candidates = ["income", "english", "calworks", "lunch", "expenditure", "computer"]
    available_controls = [c for c in control_candidates if c in df.columns]
    used_cols = base_cols + available_controls
    df_model = df[used_cols].dropna()

    print(f"Number of observations used in full model: {len(df_model)}")

    # Simple correlation for intuition.
    corr = df_model["testscr"].corr(df_model["stratio"])
    print(f"Correlation between testscr and student-teacher ratio: {corr:.3f}")

    # Full OLS model with controls.
    y = df_model["testscr"]
    X = sm.add_constant(df_model[["stratio"] + available_controls])
    model = sm.OLS(y, X).fit()

    print("\nOLS regression of testscr on student-teacher ratio and controls:")
    print(model.summary())

    beta = float(model.params["stratio"])
    p_value = float(model.pvalues["stratio"])

    print(f"\nCoefficient on stratio: {beta:.4f}")
    print(f"P-value for stratio: {p_value:.4g}")

    scalar = compute_support_scalar(beta, p_value, df_model["stratio"], df_model["testscr"])
    print(f"\nMapped Likert-style scalar ([-100, 100]): {scalar}")

    # Write scalar to conclusion.txt as the only content.
    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

