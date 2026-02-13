import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on metadata in info.json
    # feature6: total enrollment
    # feature7: number of teachers (FTE)
    # feature14: average reading score
    # feature15: average math score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["test_score_avg"] = (df["feature14"] + df["feature15"]) / 2.0

    # Basic descriptive statistics
    print("Descriptive statistics for student-teacher ratio and average test score:")
    print(df[["student_teacher_ratio", "test_score_avg"]].describe())
    print()

    # Correlation
    corr = df["student_teacher_ratio"].corr(df["test_score_avg"])
    print(f"Correlation between student-teacher ratio and average test score (all data): {corr:.4f}")
    print()

    # Simple linear regression: test_score_avg ~ student_teacher_ratio
    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(df["test_score_avg"], X).fit()

    print("OLS regression of average test score on student-teacher ratio (all data):")
    print(model.summary())
    print()

    # Repeat analysis on a restricted, more plausible band of ratios
    # to reduce the influence of extreme outliers.
    restricted = df[df["student_teacher_ratio"].between(10, 40)]
    print(
        "Number of observations with student-teacher ratio between 10 and 40:",
        len(restricted),
    )
    print(
        restricted[["student_teacher_ratio", "test_score_avg"]].describe()
    )
    print()

    if len(restricted) > 0:
        corr_restricted = restricted["student_teacher_ratio"].corr(
            restricted["test_score_avg"]
        )
        print(
            "Correlation between student-teacher ratio and average test score "
            f"(ratio between 10 and 40): {corr_restricted:.4f}"
        )
        print()

        Xr = sm.add_constant(restricted["student_teacher_ratio"])
        model_restricted = sm.OLS(restricted["test_score_avg"], Xr).fit()
        print(
            "OLS regression of average test score on student-teacher ratio "
            "(ratio between 10 and 40):"
        )
        print(model_restricted.summary())


if __name__ == "__main__":
    main()
