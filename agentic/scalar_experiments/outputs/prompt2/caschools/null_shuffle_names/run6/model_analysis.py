import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio (students per teacher).
    # Based on info.json, "english" is total enrollment and "students" is number of teachers.
    df["str_ratio"] = df["english"] / df["students"]

    # Academic performance measures: reading, math, and their average.
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    print("Student-teacher ratio summary:")
    print(df["str_ratio"].describe())

    corr = df["str_ratio"].corr(df["avg_score"])
    print(f"Correlation between student-teacher ratio and average score: {corr:.4f}")

    # Simple linear regression: avg_score ~ str_ratio
    X = sm.add_constant(df["str_ratio"])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X).fit()
    print("\nSimple OLS: avg_score ~ str_ratio")
    print(model_simple.summary())

    # Multiple regression controlling for key covariates available in the data.
    # Use income, percent CalWorks, percent reduced-price lunch, percent English learners, and expenditure per student.
    df["pct_calworks"] = df["school"]          # percent qualifying for CalWorks
    df["pct_lunch"] = df["computer"]           # percent qualifying for reduced-price lunch
    df["pct_english_learner"] = df["rownames"]  # percent English learners
    df["exp_per_student"] = df["grades"]       # expenditure per student

    predictors = [
        "str_ratio",
        "income",
        "pct_calworks",
        "pct_lunch",
        "pct_english_learner",
        "exp_per_student",
    ]

    X_multi = sm.add_constant(df[predictors])
    model_multi = sm.OLS(y, X_multi).fit()
    print("\nMultiple OLS: avg_score ~ str_ratio + controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

