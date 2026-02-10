import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio: students per teacher
    # Avoid division by zero by dropping any rows with non-positive teachers
    df = df[df["teachers"] > 0].copy()
    df["stratio"] = df["students"] / df["teachers"]

    # Define academic performance measure as the mean of reading and math scores
    if {"read", "math"}.issubset(df.columns):
        df["perf"] = (df["read"] + df["math"]) / 2.0
    else:
        raise ValueError("Expected 'read' and 'math' score columns not found in data.")

    # Drop rows with missing values in the variables of interest
    sub = df[["stratio", "perf"]].dropna()

    # Compute Pearson correlation between student-teacher ratio and performance
    r, p_value = stats.pearsonr(sub["stratio"], sub["perf"])

    # The research question asks whether a LOWER student-teacher ratio
    # is associated with HIGHER performance.
    # A negative correlation between ratio and performance supports this.
    # Map association in the direction of the question to [-100, 100]:
    # directional_corr = -r  (so stronger negative r -> larger positive score)
    directional_corr = -r

    # Clip to [-1, 1] to be safe, then scale to [-100, 100]
    directional_corr = float(np.clip(directional_corr, -1.0, 1.0))
    scalar = int(np.rint(100.0 * directional_corr))

    # Write ONLY the scalar value to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))

    # Optionally print a brief summary for interactive inspection (not written to file)
    print("Pearson r (ratio vs perf):", r)
    print("p-value:", p_value)
    print("Directional scalar (Likert -100..100):", scalar)


if __name__ == "__main__":
    main()

