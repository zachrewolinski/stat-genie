import pandas as pd
from pathlib import Path


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # According to info.json metadata:
    # - "english" is total enrollment
    # - "students" is number of teachers
    # - "district" is average reading score
    # - "expenditure" is average math score
    #
    # Student-teacher ratio (STR) and average test score.
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing or non-finite values in the key variables.
    sub = df[["stratio", "avg_score"]].replace([float("inf"), float("-inf")], pd.NA).dropna()

    if sub.empty or sub["stratio"].nunique() < 2 or sub["avg_score"].nunique() < 2:
        # Dataset does not allow us to answer; choose neutral.
        scalar = 0
    else:
        # Pearson correlation between STR and test scores.
        corr = sub["stratio"].corr(sub["avg_score"])

        # Research question: "Is a lower student-teacher ratio associated with higher academic performance?"
        # This corresponds to a *negative* correlation between STR and performance.
        #
        # Map correlation r in [-1, 1] to Likert scalar in [-100, 100] such that:
        # - Negative r (evidence that lower STR -> higher performance) yields positive scalar.
        # - Positive r (evidence against the hypothesis) yields negative scalar.
        # - Magnitude reflects strength of association.
        scalar = int(round(-corr * 100))

        # Clip to the required bounds just in case of numerical noise.
        scalar = max(-100, min(100, scalar))

    # Write the final scalar to conclusion.txt, with no extra text.
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

