import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and an overall test score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing values in the variables of interest
    df = df.dropna(
        subset=[
            "student_teacher_ratio",
            "test_score",
            "feature8",
            "feature9",
            "feature11",
            "feature12",
            "feature13",
        ]
    )

    # Simple correlation
    corr = df["student_teacher_ratio"].corr(df["test_score"])

    # Simple regression: test_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["test_score"], X_simple).fit()

    # Multiple regression with key socioeconomic controls
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X_multi = sm.add_constant(df[["student_teacher_ratio"] + controls])
    model_multi = sm.OLS(df["test_score"], X_multi).fit()

    print("Correlation between STR and test score:", corr)
    print("\nSimple regression (test_score ~ STR)")
    print(model_simple.summary())
    print("\nMultiple regression with controls")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

