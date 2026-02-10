import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: enrollment divided by number of teachers.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    ratio = df["student_teacher_ratio"]
    score = df["avg_score"]

    # Pearson correlation between student-teacher ratio and performance.
    corr = ratio.corr(score)

    # Map correlation to Likert-scale scalar answering:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    # Negative correlation implies a "Yes" (since lower ratios correspond to higher scores).
    likert_value = -corr * 100.0
    if likert_value > 100:
        likert_value = 100.0
    elif likert_value < -100:
        likert_value = -100.0

    scalar = int(round(likert_value))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

