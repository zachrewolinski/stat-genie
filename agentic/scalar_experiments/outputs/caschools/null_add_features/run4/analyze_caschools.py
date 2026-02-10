import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: more students per teacher = larger ratio
    stratio = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    performance = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in either series, if any
    valid = stratio.notna() & performance.notna()
    stratio_valid = stratio[valid]
    performance_valid = performance[valid]

    if len(stratio_valid) < 2:
        # Not enough data to assess relationship; neutral conclusion
        scalar = 0
    else:
        # Pearson correlation between ratio and performance
        r = float(stratio_valid.corr(performance_valid))

        # We want to answer: "Is a lower student-teacher ratio associated with higher performance?"
        # A negative correlation between ratio and performance supports "Yes".
        # Map correlation to [-100, 100] Likert scale, flipping sign so that
        # stronger negative r -> larger positive scalar.
        scalar = int(round(-100.0 * r))
        scalar = int(max(-100, min(100, scalar)))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

