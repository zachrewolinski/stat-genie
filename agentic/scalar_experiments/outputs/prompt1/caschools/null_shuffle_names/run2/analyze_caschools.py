import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to info.json metadata:
    # - "english" is total enrollment
    # - "students" is number of teachers
    # - "district" is average reading score
    # - "expenditure" is average math score
    #
    # Construct student-teacher ratio and an overall academic performance metric.
    df = df.copy()
    # Candidate 1: enrollment divided by number of teachers
    df["ratio_enroll_per_teacher"] = df["english"] / df["students"]
    # Candidate 2: number of students per "english" (if english were teachers)
    df["ratio_students_per_english"] = df["students"] / df["english"]
    df["test_score_avg"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop any rows with missing or invalid values
    df = df.replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    df = df.dropna(
        subset=["ratio_enroll_per_teacher", "ratio_students_per_english", "test_score_avg"]
    )

    # Basic descriptive statistics
    print("Number of districts:", len(df))
    print("\nDescriptive statistics:")
    print(df[["ratio_enroll_per_teacher", "ratio_students_per_english", "test_score_avg"]].describe())

    # Correlation between candidate ratios and average test score
    corr1 = df["ratio_enroll_per_teacher"].corr(df["test_score_avg"])
    corr2 = df["ratio_students_per_english"].corr(df["test_score_avg"])
    print("\nPearson correlation (enroll/teacher vs. test score):", corr1)
    print("Pearson correlation (students/english vs. test score):", corr2)

    # Simple linear regression: test_score_avg ~ student_teacher_ratio
    # For regression, treat ratio_enroll_per_teacher as the working
    # definition of student-teacher ratio (enrollment per teacher).
    X = sm.add_constant(df["ratio_enroll_per_teacher"])
    y = df["test_score_avg"]
    model = sm.OLS(y, X).fit()

    print("\nOLS regression results:")
    print(model.summary())

    coef = model.params["ratio_enroll_per_teacher"]
    p_value = model.pvalues["ratio_enroll_per_teacher"]
    print("\nSlope for ratio_enroll_per_teacher:", coef)
    print("p-value for slope:", p_value)


if __name__ == "__main__":
    main()
