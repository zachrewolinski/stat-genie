import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map columns using the metadata in info.json
    enrollment = df["feature6"]
    teachers = df["feature7"]
    calworks = df["feature8"]
    lunch = df["feature9"]
    computer = df["feature10"]
    expenditure = df["feature11"]
    income = df["feature12"]
    english = df["feature13"]
    read = df["feature14"]
    math = df["feature15"]

    # Construct key variables
    student_teacher_ratio = enrollment / teachers
    avg_score = (read + math) / 2.0

    data = pd.DataFrame(
        {
            "avg_score": avg_score,
            "stratio": student_teacher_ratio,
            "calworks": calworks,
            "lunch": lunch,
            "computer": computer,
            "expenditure": expenditure,
            "income": income,
            "english": english,
        }
    ).dropna()

    # Simple bivariate regression: avg_score ~ student-teacher ratio
    X_simple = sm.add_constant(data["stratio"])
    model_simple = sm.OLS(data["avg_score"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    X_controls = data[
        [
            "stratio",
            "calworks",
            "lunch",
            "computer",
            "expenditure",
            "income",
            "english",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(data["avg_score"], X_controls).fit()

    print("Simple regression: avg_score ~ student-teacher ratio")
    print("Coefficient on stratio:", model_simple.params["stratio"])
    print("p-value:", model_simple.pvalues["stratio"])
    print("R-squared:", model_simple.rsquared)
    print()

    print("Multiple regression with controls")
    print("Coefficient on stratio:", model_controls.params["stratio"])
    print("p-value:", model_controls.pvalues["stratio"])
    print("R-squared:", model_controls.rsquared)


if __name__ == "__main__":
    main()

