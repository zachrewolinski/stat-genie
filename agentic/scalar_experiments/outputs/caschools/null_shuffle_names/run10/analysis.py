import numpy as np
import pandas as pd
from scipy import stats


def load_data(path: str = "caschools.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # According to info.json, the true semantics are:
    # - english  -> total enrollment (students)
    # - students -> number of teachers
    # - district -> average reading score
    # - expenditure -> average math score
    #
    # Construct student–teacher ratio and average achievement.
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    # Drop any rows with missing or zero teachers to avoid invalid ratios.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "avg_score"])
    df = df[df["students"] != 0]

    return df


def compute_likert_scalar(corr: float, pval: float) -> int:
    """
    Map the correlation and its p-value to a [-100, 100] Likert-style scalar.

    Positive values support the research hypothesis:
      "Lower student–teacher ratios are associated with higher achievement."
    Negative values contradict it. Zero is neutral / no clear association.
    """
    if np.isnan(corr) or np.isnan(pval):
        return 0

    # If correlation is essentially zero, treat as neutral.
    if abs(corr) < 0.05:
        return 0

    # A negative correlation means higher STR (more students per teacher)
    # is associated with lower achievement, which supports the hypothesis.
    # So we flip the sign: negative corr -> positive scalar.
    if corr < 0:
        direction_sign = 1
    elif corr > 0:
        direction_sign = -1
    else:
        return 0

    # Scale the strength of association by the correlation magnitude.
    # Typical substantial correlations are in |r| ∈ [0, 0.5]; cap at 0.5.
    corr_strength = min(1.0, abs(corr) / 0.5)

    # Weight by statistical significance.
    if pval < 1e-6:
        sig_weight = 1.0
    elif pval < 1e-3:
        sig_weight = 0.9
    elif pval < 1e-2:
        sig_weight = 0.8
    elif pval < 5e-2:
        sig_weight = 0.6
    elif pval < 1e-1:
        sig_weight = 0.4
    else:
        sig_weight = 0.2

    strength = corr_strength * sig_weight
    scalar = int(round(direction_sign * strength * 100))
    scalar = max(-100, min(100, scalar))
    return scalar


def main() -> None:
    df = load_data()

    # Pearson correlation between student–teacher ratio and average achievement.
    corr, pval = stats.pearsonr(df["stratio"], df["avg_score"])

    scalar = compute_likert_scalar(corr, pval)

    # Print a brief summary for human inspection (stdout is unrestricted).
    print("Number of districts:", len(df))
    print("Mean student–teacher ratio:", df["stratio"].mean())
    print("Mean achievement score:", df["avg_score"].mean())
    print("Correlation (STR vs achievement):", corr)
    print("P-value:", pval)
    print("Derived Likert scalar ([-100, 100]):", scalar)

    # Write the scalar conclusion as required.
    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

