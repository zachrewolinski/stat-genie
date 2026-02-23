import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: total enrollment divided by number of teachers.
    df["stud_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing values in variables of interest (there should be none, but be safe).
    df_model = df.dropna(
        subset=[
            "stud_teacher_ratio",
            "avg_score",
            "feature8",
            "feature9",
            "feature10",
            "feature11",
            "feature12",
            "feature13",
        ]
    ).copy()

    print("Number of districts in model:", len(df_model))

    # Simple correlation between ratio and average score.
    corr = df_model["stud_teacher_ratio"].corr(df_model["avg_score"])
    print("Correlation (stud_teacher_ratio, avg_score):", corr)

    # Simple OLS: avg_score ~ stud_teacher_ratio
    X_simple = sm.add_constant(df_model["stud_teacher_ratio"])
    y = df_model["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nSimple OLS: avg_score ~ stud_teacher_ratio")
    print(model_simple.summary())

    # Multiple OLS controlling for key covariates related to resources and demographics.
    covariates = [
        "stud_teacher_ratio",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    X_multi = sm.add_constant(df_model[covariates])
    model_multi = sm.OLS(y, X_multi).fit()
    print("\nMultiple OLS: avg_score ~ stud_teacher_ratio + controls")
    print(model_multi.summary())

    # Print key quantities we will use for the conclusion.
    print("\nKey results:")
    print("Simple OLS coef (stud_teacher_ratio):", model_simple.params["stud_teacher_ratio"])
    print("Simple OLS p-value (stud_teacher_ratio):", model_simple.pvalues["stud_teacher_ratio"])
    print("Simple OLS R-squared:", model_simple.rsquared)
    print("Multiple OLS coef (stud_teacher_ratio):", model_multi.params["stud_teacher_ratio"])
    print("Multiple OLS p-value (stud_teacher_ratio):", model_multi.pvalues["stud_teacher_ratio"])
    print("Multiple OLS R-squared:", model_multi.rsquared)


if __name__ == "__main__":
    main()

