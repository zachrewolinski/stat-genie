import pandas as pd
import numpy as np


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Construct key variables using metadata from info.json:
    # - english: total enrollment
    # - students: number of teachers
    # Thus student-teacher ratio = students per teacher = enrollment / teachers.
    # - district: average reading score
    # - expenditure: average math score
    # We treat academic performance as the average of reading and math scores.

    df = df.copy()

    # Guard against division by zero or missing values
    valid_teachers = df["students"].replace(0, np.nan)
    df["student_teacher_ratio"] = df["english"] / valid_teachers

    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing values in the variables of interest
    subset = df[["student_teacher_ratio", "avg_score"]].dropna()

    if subset.empty:
        raise ValueError("No valid observations for student_teacher_ratio and avg_score.")

    # Pearson correlation between student-teacher ratio and academic performance
    corr = subset["student_teacher_ratio"].corr(subset["avg_score"])

    # Map correlation into a Likert-style scalar:
    # Positive scalar => evidence that LOWER ratio (more teachers per student)
    # is associated with HIGHER academic performance.
    # Since higher ratios mean more students per teacher, we negate the correlation.
    scalar = -corr * 100.0

    # Clip to [-100, 100] and round to nearest integer
    scalar = int(np.clip(np.rint(scalar), -100, 100))

    # Write the scalar to conclusion.txt with no extra text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

