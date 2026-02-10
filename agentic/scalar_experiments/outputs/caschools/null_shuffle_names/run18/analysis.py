import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Map columns to their semantic meaning based on info.json descriptions
    enrollment = df["english"]  # Total enrollment
    teachers = df["students"]  # Number of teachers
    read_score = df["district"]  # Average reading score
    math_score = df["expenditure"]  # Average math score

    # Construct key variables
    student_teacher_ratio = enrollment / teachers
    test_score = (read_score + math_score) / 2.0

    data = pd.DataFrame(
        {
            "stratio": student_teacher_ratio,
            "testscr": test_score,
            "read": read_score,
            "math": math_score,
        }
    ).dropna()

    # Basic descriptive statistics
    desc = data.describe()

    # Correlations
    corr_testscr = data["stratio"].corr(data["testscr"])
    corr_read = data["stratio"].corr(data["read"])
    corr_math = data["stratio"].corr(data["math"])

    # Simple linear regression: testscr ~ stratio
    X = sm.add_constant(data["stratio"])
    y = data["testscr"]
    model = sm.OLS(y, X).fit()

    print("Descriptive statistics for key variables:")
    print(desc)
    print("\nCorrelations between student-teacher ratio and scores:")
    print(f"  corr(stratio, testscr) = {corr_testscr:.3f}")
    print(f"  corr(stratio, read)    = {corr_read:.3f}")
    print(f"  corr(stratio, math)    = {corr_math:.3f}")

    print("\nOLS regression: testscr ~ stratio")
    print(model.summary())


if __name__ == "__main__":
    main()

