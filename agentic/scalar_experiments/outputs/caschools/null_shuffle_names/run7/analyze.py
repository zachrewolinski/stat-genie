import pandas as pd
import numpy as np


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # According to the metadata, 'english' is total enrollment
    # and 'students' is the number of teachers.
    # Use these to construct the student–teacher ratio.
    df["student_teacher_ratio"] = df["english"] / df["students"]

    # Academic performance: use the average of reading and math scores.
    # Metadata indicates 'district' = average reading score
    # and 'expenditure' = average math score.
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any rows with missing values in the key variables
    subset = df[["student_teacher_ratio", "avg_score"]].dropna()

    # Compute Pearson correlation between ratio and performance
    if len(subset) < 2:
        # Fallback: if we somehow have too few observations,
        # report a neutral conclusion.
        scalar = 0
    else:
        r = subset["student_teacher_ratio"].corr(subset["avg_score"])

        # Map correlation to Likert scale:
        # - Negative correlation (lower ratio -> higher scores) should
        #   yield a positive scalar (strong "Yes").
        # - Positive correlation (higher ratio -> higher scores) should
        #   yield a negative scalar (strong "No").
        scalar_float = -100.0 * r
        # Round to nearest integer and clip to [-100, 100]
        scalar = int(np.clip(np.round(scalar_float), -100, 100))

    # Write scalar conclusion to the required file
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

