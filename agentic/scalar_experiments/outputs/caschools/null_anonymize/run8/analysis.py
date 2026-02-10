import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on info.json metadata.
    enrollment = df["feature6"]
    teachers = df["feature7"]
    read_score = df["feature14"]
    math_score = df["feature15"]

    # Student–teacher ratio: students per teacher.
    df["student_teacher_ratio"] = enrollment / teachers

    # Overall academic performance: average of reading and math scores.
    df["test_score"] = (read_score + math_score) / 2.0

    # Basic correlation.
    corr = df["student_teacher_ratio"].corr(df["test_score"])

    # Simple linear regression of test scores on student–teacher ratio.
    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(df["test_score"], X, missing="drop").fit()

    print("Number of observations:", int(model.nobs))
    print("Correlation (ratio, test_score):", float(corr))
    print("OLS coefficient on student_teacher_ratio:", float(model.params["student_teacher_ratio"]))
    print("OLS t-stat for student_teacher_ratio:", float(model.tvalues["student_teacher_ratio"]))
    print("OLS p-value for student_teacher_ratio:", float(model.pvalues["student_teacher_ratio"]))
    print("R-squared:", float(model.rsquared))


if __name__ == "__main__":
    main()

