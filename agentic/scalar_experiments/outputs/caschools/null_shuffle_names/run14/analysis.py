import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json descriptions:
    # - "english" is total enrollment
    # - "students" is number of teachers
    # - "district" is average reading score
    # - "expenditure" is average math score
    df = df.copy()
    df["enrollment"] = df["english"]
    df["n_teachers"] = df["students"]

    # Guard against division by zero or missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["enrollment", "n_teachers", "district", "expenditure"])
    df = df[df["n_teachers"] != 0]

    # Student-teacher ratio: students per teacher
    df["stratio"] = df["enrollment"] / df["n_teachers"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Compute Pearson correlation between ratio and test scores
    r, p_value = stats.pearsonr(df["stratio"], df["testscr"])

    # Map correlation to Likert scale:
    # - Negative correlation (more students per teacher -> lower scores)
    #   supports the research claim that a lower ratio improves performance.
    # - We flip the sign so stronger negative r -> stronger "Yes".
    # - Clip to [-100, 100] and round to nearest integer.
    raw_score = -100 * r
    likert_score = int(np.clip(np.round(raw_score), -100, 100))

    # For transparency while debugging, print summary statistics
    print("N:", len(df))
    print("Pearson r (stratio, testscr):", r)
    print("p-value:", p_value)
    print("Mapped Likert score:", likert_score)

    # Write final scalar conclusion
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert_score))


if __name__ == "__main__":
    main()

