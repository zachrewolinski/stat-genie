import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def compute_likert_scalar(corr: float, p_value: float) -> int:
    """
    Map the correlation and p-value to a Likert-style integer in [-100, 100].

    Negative correlation between student-teacher ratio and performance
    supports the research question ("lower ratio -> higher performance").
    """
    if np.isnan(corr) or np.isnan(p_value):
        return 0

    # Strength of association from |corr|; 0.5 treated as "strong"
    strength = min(1.0, abs(corr) / 0.5)

    # Significance weight based on p-value thresholds
    if p_value >= 0.10:
        significance = 0.0
    elif p_value >= 0.05:
        significance = 0.25
    elif p_value >= 0.01:
        significance = 0.5
    elif p_value >= 0.001:
        significance = 0.75
    else:
        significance = 1.0

    evidence = strength * significance

    if corr < 0:
        scalar = int(round(100 * evidence))
    else:
        scalar = int(round(-100 * evidence))

    return max(-100, min(100, scalar))


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio = total enrollment / number of teachers
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["academic_performance"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop rows with missing values in key variables
    subset = df[["student_teacher_ratio", "academic_performance"]].dropna()

    # Compute Pearson correlation and p-value
    corr, p_value = pearsonr(
        subset["student_teacher_ratio"], subset["academic_performance"]
    )

    scalar = compute_likert_scalar(corr, p_value)

    # Write the scalar conclusion to the required file
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

