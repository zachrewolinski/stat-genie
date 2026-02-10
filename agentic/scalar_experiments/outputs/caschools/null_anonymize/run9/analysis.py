import numpy as np
import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: total enrollment divided by number of teachers.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores.
    df["avg_test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing values in the key variables (defensive; none expected).
    sub = df[["student_teacher_ratio", "avg_test_score"]].dropna()

    if sub.empty:
        likert_score = 0
    else:
        # Pearson correlation between student-teacher ratio and test scores.
        r = sub["student_teacher_ratio"].corr(sub["avg_test_score"])

        if pd.isna(r):
            likert_score = 0
        else:
            # Lower student-teacher ratio (smaller value) associated with higher performance
            # corresponds to a negative correlation. Map this to a positive Likert score.
            likert_score = int(round(-100.0 * r))
            likert_score = max(-100, min(100, likert_score))

    # Write the scalar conclusion to the required file with no extra text.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert_score))


if __name__ == "__main__":
    main()

