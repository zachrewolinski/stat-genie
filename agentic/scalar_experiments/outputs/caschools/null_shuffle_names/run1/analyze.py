import math
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning using info.json:
    # english -> total enrollment
    # students -> number of teachers
    # district -> average reading score
    # expenditure -> average math score
    enrollment = df["english"].astype(float)
    teachers = df["students"].astype(float)
    reading = df["district"].astype(float)
    math_score = df["expenditure"].astype(float)

    # Compute student-teacher ratio (students per teacher)
    stratio = enrollment / teachers

    # Overall academic performance as the mean of reading and math scores
    performance = (reading + math_score) / 2.0

    # Compute Pearson correlation between student-teacher ratio and performance
    corr_matrix = np.corrcoef(stratio, performance)
    corr = float(corr_matrix[0, 1])

    # Translate correlation into a Likert-style scalar answering:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    #
    # Negative correlation means: higher ratio -> lower performance,
    # which supports the research question (lower ratio => higher performance).
    # Map this to [-100, 100] by flipping the sign so positive values indicate support.
    scalar = -corr * 100.0

    # Clip to the valid Likert range and round to the nearest integer
    scalar = max(-100.0, min(100.0, scalar))
    scalar_int = int(round(scalar))

    # Write the conclusion scalar to conclusion.txt
    out_path = Path("conclusion.txt")
    out_path.write_text(str(scalar_int), encoding="utf-8")


if __name__ == "__main__":
    main()

