import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning using info.json descriptions.
    # english  -> total enrollment (students)
    # students -> number of teachers
    # district -> average reading score
    # expenditure -> average math score
    enrollment = pd.to_numeric(df["english"], errors="coerce")
    teachers = pd.to_numeric(df["students"], errors="coerce")
    read_score = pd.to_numeric(df["district"], errors="coerce")
    math_score = pd.to_numeric(df["expenditure"], errors="coerce")

    # Compute student-teacher ratio and overall academic performance
    with np.errstate(divide="ignore", invalid="ignore"):
        stratio = enrollment / teachers
    tests = (read_score + math_score) / 2.0

    # Drop any rows with missing or non-finite values
    mask = (
        np.isfinite(stratio)
        & np.isfinite(tests)
    )
    stratio_clean = stratio[mask]
    tests_clean = tests[mask]

    # If we have too few observations, treat evidence as neutral
    if stratio_clean.size < 3:
        likert_scalar = 0
    else:
        # Estimate Pearson correlation between student-teacher ratio and test scores
        r, _ = stats.pearsonr(stratio_clean, tests_clean)

        # Transform correlation into a Likert-style scalar on [-100, 100].
        # Negative correlation means lower ratios (smaller classes) are associated
        # with higher performance, which supports the research question.
        likert_scalar = int(round(np.clip(-100.0 * r, -100.0, 100.0)))

    # Write scalar conclusion to file with no extra text or newline
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert_scalar))


if __name__ == "__main__":
    main()

