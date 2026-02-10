import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Focus on core variables for the research question.
    df = df.dropna(subset=["students", "teachers", "read", "math"])
    df = df[df["teachers"] > 0]

    # Student-teacher ratio (higher = larger classes).
    df["student_teacher_ratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Pearson correlation between ratio and scores.
    correlation = df["student_teacher_ratio"].corr(df["avg_score"])

    # Map correlation to Likert scale where:
    # - Negative correlation (smaller ratio -> higher scores) supports "Yes".
    # - Positive correlation supports "No".
    likert_score = int(round(-100 * correlation))
    likert_score = max(-100, min(100, likert_score))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert_score))


if __name__ == "__main__":
    main()

