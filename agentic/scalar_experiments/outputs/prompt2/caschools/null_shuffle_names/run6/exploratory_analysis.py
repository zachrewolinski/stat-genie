import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Inspect basic structure
    print("Columns:", df.columns.tolist())
    print(df.head())
    print(df.describe(include="all"))

    # Derive plausible student-teacher ratio candidates
    # Based on info.json, "english" ~ total enrollment, "students" ~ number of teachers.
    df["str_enroll_teachers"] = df["english"] / df["students"]

    # Also try using "lunch" as enrollment proxy and "english" as teachers, just in case.
    df["str_lunch_english"] = df["lunch"] / df["english"]

    print(df[["english", "students", "str_enroll_teachers"]].describe())
    print(df[["lunch", "english", "str_lunch_english"]].describe())

    # For academic performance, use reading, math, and their average.
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    print(df[["read_score", "math_score", "avg_score"]].describe())

    # Compute simple correlations
    corr_matrix = df[["str_enroll_teachers", "str_lunch_english", "read_score", "math_score", "avg_score"]].corr()
    print("Correlation matrix:")
    print(corr_matrix)


if __name__ == "__main__":
    main()

