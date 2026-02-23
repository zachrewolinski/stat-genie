import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables following standard CASchools usage.
    df["students_per_teacher"] = df["feature6"] / df["feature7"]
    df["test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing values in variables used below (should be none, but be safe).
    vars_simple = ["test_score", "students_per_teacher"]
    vars_controls = vars_simple + [
        "feature8",   # percent CalWorks
        "feature9",   # percent reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # percent English learners
    ]

    df_simple = df[vars_simple].dropna()
    df_controls = df[vars_controls].dropna()

    # Simple bivariate regression: test score on students_per_teacher.
    X_simple = sm.add_constant(df_simple["students_per_teacher"])
    y_simple = df_simple["test_score"]
    model_simple = sm.OLS(y_simple, X_simple).fit()

    # Multiple regression with standard socioeconomic controls.
    X_controls = sm.add_constant(
        df_controls[
            [
                "students_per_teacher",
                "feature8",
                "feature9",
                "feature11",
                "feature12",
                "feature13",
            ]
        ]
    )
    y_controls = df_controls["test_score"]
    model_controls = sm.OLS(y_controls, X_controls).fit()

    print("Simple regression: test_score ~ students_per_teacher")
    print(model_simple.summary())
    print()
    print("Controlled regression: test_score ~ students_per_teacher + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

