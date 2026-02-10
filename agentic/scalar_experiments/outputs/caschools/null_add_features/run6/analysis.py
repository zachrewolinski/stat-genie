import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (defensive, though none expected)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "testscr"])

    # Simple bivariate OLS: testscr ~ stratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()

    slope = float(model.params["stratio"])
    pval = float(model.pvalues["stratio"])

    # Pearson correlation between student-teacher ratio and test scores
    r = float(df["stratio"].corr(df["testscr"]))

    # Map evidence to Likert scalar in [-100, 100]
    scalar = evidence_to_scalar(r, pval)

    # Write only the scalar to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


def evidence_to_scalar(r: float, pval: float) -> int:
    """
    Convert correlation and p-value into a Likert-style scalar:
    -100 (very strong 'No') to 100 (very strong 'Yes').

    Here "Yes" means: lower student-teacher ratios are associated with higher
    academic performance, which corresponds to a *negative* correlation between
    ratio and test scores.
    """
    # No or very weak evidence: stay near neutral
    if math.isnan(r) or math.isnan(pval) or abs(r) < 0.05 or pval >= 0.10:
        return 0

    # Direction: negative r supports "Yes", positive r supports "No"
    direction = -1.0 if r > 0 else 1.0

    # Strength from effect size (saturate around |r| = 0.5)
    effect_strength = min(1.0, abs(r) / 0.5)

    # Strength from statistical significance
    if pval < 1e-6:
        p_strength = 1.0
    elif pval < 1e-3:
        p_strength = 0.9
    elif pval < 1e-2:
        p_strength = 0.7
    else:
        p_strength = 0.5

    raw = 100.0 * direction * effect_strength * p_strength
    # Clamp to [-100, 100] and round to nearest integer
    scalar = int(round(max(-100.0, min(100.0, raw))))
    return scalar


if __name__ == "__main__":
    main()

