import pandas as pd
import numpy as np
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Core variables
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["students", "teachers", "read", "math"])

    # Students per teacher (higher = larger classes)
    df["str"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Drop any remaining NAs in key variables
    df = df.dropna(subset=["str", "score"])

    # Pearson correlation between class size (str) and performance
    r, p = stats.pearsonr(df["str"], df["score"])

    # We interpret the research question as:
    #   Lower student-teacher ratio (smaller str) -> higher performance
    # That corresponds to a negative correlation between str and score.

    # Map effect size and significance to a Likert-scale scalar [-100, 100].
    def compute_scalar(r_value: float, p_value: float) -> int:
        # If association is very weak or clearly non-significant, return neutral.
        if np.isnan(r_value) or np.isnan(p_value):
            return 0

        abs_r = abs(r_value)

        # No evidence: tiny effect or very high p-value
        if abs_r < 0.05 or p_value > 0.5:
            base = 0
        elif abs_r < 0.15:
            base = 25
        elif abs_r < 0.3:
            base = 50
        elif abs_r < 0.5:
            base = 75
        else:
            base = 100

        # Direction relative to the research question:
        # Negative r supports "yes" (lower ratio -> higher performance).
        if r_value < 0:
            return int(base)
        elif r_value > 0:
            return int(-base)
        else:
            return 0

    scalar = compute_scalar(r, p)

    # Ensure scalar is clipped to [-100, 100]
    scalar = int(max(-100, min(100, scalar)))

    # Write scalar conclusion to file with no extra text.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

