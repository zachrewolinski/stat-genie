import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def compute_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Student-teacher ratio: total enrollment / number of teachers
    df = df.copy()
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)
    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    """
    Analyze association between student-teacher ratio and academic performance.

    Returns a dict with correlation and a scalar conclusion in [-100, 100].
    """
    ratio = df["student_teacher_ratio"]
    score = df["avg_score"]

    corr = ratio.corr(score)

    # Map correlation to Likert-style scalar answering:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    #
    # Negative correlation means higher ratios (more students per teacher)
    # are associated with lower scores, i.e., evidence for "Yes".
    # We treat |corr| = 0.5 or higher as very strong evidence (|scalar|=100),
    # and scale linearly below that.
    max_ref = 0.5
    strength = min(abs(corr) / max_ref, 1.0)

    if np.isnan(corr) or strength == 0:
        scalar = 0
    else:
        if corr < 0:
            # Negative correlation -> evidence that lower ratio helps -> positive scalar
            sign = 1
        else:
            # Positive correlation -> evidence against the hypothesis -> negative scalar
            sign = -1
        scalar = int(round(sign * strength * 100))

    return {
        "correlation": float(corr) if not np.isnan(corr) else np.nan,
        "scalar": scalar,
    }


def write_conclusion(scalar: int, path: str = "conclusion.txt") -> None:
    Path(path).write_text(f"{int(scalar)}", encoding="utf-8")


def main() -> None:
    # Load dataset
    csv_path = "caschools.csv"
    df = load_data(csv_path)

    # Construct relevant variables
    df = compute_variables(df)

    # Analyze relationship and derive scalar
    results = analyze_relationship(df)

    # Print a brief analysis summary for the user
    corr = results["correlation"]
    scalar = results["scalar"]

    print("Correlation between student-teacher ratio and average score:", corr)
    print("Scalar conclusion (Likert -100 to 100):", scalar)

    # Write scalar to conclusion.txt as required
    write_conclusion(scalar)


if __name__ == "__main__":
    main()

