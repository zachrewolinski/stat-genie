import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on the metadata description
    # feature6: total enrollment
    # feature7: number of teachers
    # feature14: average reading score
    # feature15: average math score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Basic correlation
    corr = df["student_teacher_ratio"].corr(df["avg_testscr"])

    # Simple OLS: average score on student-teacher ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["avg_testscr"], X_simple).fit()

    # Multiple regression with key covariates from the metadata
    covariates = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    X_full = sm.add_constant(df[["student_teacher_ratio"] + covariates])
    model_full = sm.OLS(df["avg_testscr"], X_full).fit()

    coef_simple = float(model_simple.params["student_teacher_ratio"])
    p_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    coef_full = float(model_full.params["student_teacher_ratio"])
    p_full = float(model_full.pvalues["student_teacher_ratio"])
    r2_full = float(model_full.rsquared)

    # Print key results so they can be inspected from the shell.
    summary = {
        "n_obs": int(df.shape[0]),
        "corr_student_teacher_ratio_avg_testscr": float(corr),
        "simple_model": {
            "coef_ratio": coef_simple,
            "p_value_ratio": p_simple,
            "r_squared": r2_simple,
        },
        "full_model": {
            "coef_ratio": coef_full,
            "p_value_ratio": p_full,
            "r_squared": r2_full,
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

