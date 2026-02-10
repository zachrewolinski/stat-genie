import pathlib

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    data_path = pathlib.Path("caschools.csv")
    df = pd.read_csv(data_path)

    # According to the metadata, the columns are relabeled, but descriptions
    # indicate that:
    # - "english" holds total enrollment
    # - "students" holds the number of teachers
    # - "district" is the average reading score
    # - "expenditure" is the average math score
    enrollment = df["english"].astype(float)
    teachers = df["students"].astype(float)

    # Construct student–teacher ratio (students per teacher).
    str_value = enrollment / teachers

    # Academic performance: average of reading and math scores.
    read_score = df["district"].astype(float)
    math_score = df["expenditure"].astype(float)
    testscr = (read_score + math_score) / 2.0

    # Drop any rows with missing or non-finite values.
    mask_finite = np.isfinite(str_value) & np.isfinite(testscr)
    str_clean = str_value[mask_finite]
    testscr_clean = testscr[mask_finite]

    # Guard against pathological cases.
    if len(str_clean) < 3:
        # With too few observations, we have essentially no information.
        scalar = 0
    else:
        # Compute Pearson correlation between the student–teacher ratio
        # and test scores.
        r, _p = stats.pearsonr(str_clean, testscr_clean)

        # Map correlation to Likert scale:
        # - Negative correlation (higher ratio -> lower scores) supports
        #   the statement that lower ratios improve performance.
        # - Positive correlation supports the opposite.
        scalar = int(round(-100.0 * r))
        scalar = max(-100, min(100, scalar))

    conclusion_path = pathlib.Path("conclusion.txt")
    conclusion_path.write_text(f"{scalar}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

