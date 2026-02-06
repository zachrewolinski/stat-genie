import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio and average academic performance
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    corr = df["student_teacher_ratio"].corr(df["avg_score"])

    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(df["avg_score"], X).fit()

    print("Correlation (STR vs avg_score):", corr)
    print(model.summary())


if __name__ == "__main__":
    main()
