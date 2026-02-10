import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Ensure numeric types where expected (dataset has some columns read as object)
    numeric_cols = [
        "students",     # number of teachers
        "english",      # total enrollment
        "district",     # average reading score
        "expenditure",  # average math score
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Student-teacher ratio: enrollment per teacher (higher = larger classes)
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: average of reading and math score variables
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Simple correlation
    corr = df["stratio"].corr(df["avg_score"])

    # Linear regression of avg_score on stratio (controls omitted to keep model simple and transparent)
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["avg_score"], X, missing="drop").fit()
    slope = float(model.params["stratio"])

    # Combine evidence:
    # - corr < 0 and slope < 0 -> lower ratio associated with higher performance.
    # - Map effect size / consistency to Likert score.
    evidence_strength = 0.0

    if corr < 0 and slope < 0:
        # Start from positive association strength based on absolute correlation
        base = min(abs(corr), 1.0)
        # Scale slope relative to mean score to normalize units
        mean_score = df["avg_score"].mean()
        rel_slope = abs(slope) / max(mean_score, 1e-6)
        # Cap relative slope to avoid extreme values
        rel_slope = min(rel_slope, 0.2)

        evidence_strength = 0.6 * base + 0.4 * (rel_slope / 0.2)
    elif corr > 0 and slope > 0:
        base = min(abs(corr), 1.0)
        mean_score = df["avg_score"].mean()
        rel_slope = abs(slope) / max(mean_score, 1e-6)
        rel_slope = min(rel_slope, 0.2)
        evidence_strength = -1.0 * (0.6 * base + 0.4 * (rel_slope / 0.2))
    else:
        # Mixed or weak evidence
        evidence_strength = 0.0

    # Map evidence_strength in [-1, 1] to Likert [-100, 100]
    scalar = int(round(max(-1.0, min(1.0, evidence_strength)) * 100))

    # Save scalar conclusion only
    Path("conclusion.txt").write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()
