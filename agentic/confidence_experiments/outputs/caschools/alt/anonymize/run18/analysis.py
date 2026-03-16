import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["testscr_avg"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing values in variables of interest (defensive)
    cols = [
        "student_teacher_ratio",
        "testscr_avg",
        "feature12",  # income
        "feature8",  # CalWorks %
        "feature9",  # lunch %
        "feature13",  # English learners %
    ]
    df_model = df[cols].dropna()

    # Simple correlation
    r, p_r = stats.pearsonr(df_model["student_teacher_ratio"], df_model["testscr_avg"])

    # Bivariate OLS: testscr_avg ~ student_teacher_ratio
    X_simple = sm.add_constant(df_model["student_teacher_ratio"])
    model_simple = sm.OLS(df_model["testscr_avg"], X_simple).fit()

    # Multivariate OLS with basic demographic controls
    control_cols = ["feature12", "feature8", "feature9", "feature13"]
    X_multi = sm.add_constant(
        df_model[["student_teacher_ratio"] + control_cols]
    )
    model_multi = sm.OLS(df_model["testscr_avg"], X_multi).fit()

    results = {
        "n_obs": int(df_model.shape[0]),
        "correlation": {
            "r_student_teacher_testscr": float(r),
            "p_value": float(p_r),
        },
        "bivariate_ols": {
            "coef_str": float(model_simple.params["student_teacher_ratio"]),
            "p_value_str": float(model_simple.pvalues["student_teacher_ratio"]),
            "r_squared": float(model_simple.rsquared),
        },
        "multivariate_ols": {
            "coef_str": float(model_multi.params["student_teacher_ratio"]),
            "p_value_str": float(model_multi.pvalues["student_teacher_ratio"]),
            "r_squared": float(model_multi.rsquared),
        },
    }

    # Print a JSON summary so we can inspect it from the CLI
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

