import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on metadata in info.json:
    # feature6: total enrollment
    # feature7: number of teachers
    # feature14: average reading score
    # feature15: average math score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["teachers_per_student"] = df["feature7"] / df["feature6"]
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Simple correlations between ratio measures and academic performance
    corr_str = df["student_teacher_ratio"].corr(df["avg_score"])
    corr_tps = df["teachers_per_student"].corr(df["avg_score"])

    # Linear regression of academic performance on student-teacher ratio
    # plus several key covariates (demographics and resources).
    X = df[
        ["student_teacher_ratio", "feature8", "feature9", "feature11", "feature12", "feature13"]
    ]
    X = sm.add_constant(X)
    y = df["avg_score"]

    model = sm.OLS(y, X).fit()

    results = {
        "corr_student_teacher_ratio_avg_score": float(corr_str),
        "corr_teachers_per_student_avg_score": float(corr_tps),
        "coef_student_teacher_ratio": float(model.params["student_teacher_ratio"]),
        "pval_student_teacher_ratio": float(model.pvalues["student_teacher_ratio"]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
    }

    # Print a compact JSON summary so it can be inspected from the shell.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
