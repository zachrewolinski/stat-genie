import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)

    # According to info.json:
    # - english: Total enrollment
    # - students: Number of teachers
    # - district: Average reading score
    # - expenditure: Average math score
    #
    # We consider two possible definitions of the ratio and
    # inspect their distributions to verify which is plausible.
    df["ratio_enroll_over_teachers"] = df["english"] / df["students"]
    df["ratio_teachers_over_enroll"] = df["students"] / df["english"]

    # Define academic performance as the average of reading and math scores.
    df["avg_testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing values in key variables (should be none, but stay safe).
    df_model = df[
        [
            "ratio_enroll_over_teachers",
            "avg_testscr",
            "district",
            "expenditure",
            "income",
            "school",
            "computer",
            "rownames",
            "grades",
            "english",
        ]
    ].dropna()

    # Simple Pearson correlations (bivariate)
    corr_testscr = df_model["ratio_enroll_over_teachers"].corr(df_model["avg_testscr"])
    corr_read = df_model["ratio_enroll_over_teachers"].corr(df_model["district"])
    corr_math = df_model["ratio_enroll_over_teachers"].corr(df_model["expenditure"])

    # Simple OLS: avg_testscr ~ student_teacher_ratio
    X_simple = sm.add_constant(df_model["ratio_enroll_over_teachers"])
    y = df_model["avg_testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multivariate OLS with key controls:
    # income, percent CalWorks (school), percent reduced-price lunch (computer),
    # percent English learners (rownames), expenditure per student (grades),
    # log enrollment (english).
    X_controls = df_model[
        ["ratio_enroll_over_teachers", "income", "school", "computer", "rownames", "grades"]
    ].copy()
    X_controls["log_enrollment"] = np.log(df_model["english"])
    X_multi = sm.add_constant(X_controls)
    model_multi = sm.OLS(y, X_multi).fit()

    summary = {
        "n_obs": int(df_model.shape[0]),
        "ratio_enroll_over_teachers_mean": float(df["ratio_enroll_over_teachers"].mean()),
        "ratio_enroll_over_teachers_min": float(df["ratio_enroll_over_teachers"].min()),
        "ratio_enroll_over_teachers_max": float(df["ratio_enroll_over_teachers"].max()),
        "ratio_teachers_over_enroll_mean": float(df["ratio_teachers_over_enroll"].mean()),
        "ratio_teachers_over_enroll_min": float(df["ratio_teachers_over_enroll"].min()),
        "ratio_teachers_over_enroll_max": float(df["ratio_teachers_over_enroll"].max()),
        "corr_ratio_enroll_over_teachers_avg_testscr": float(corr_testscr),
        "corr_ratio_enroll_over_teachers_read": float(corr_read),
        "corr_ratio_enroll_over_teachers_math": float(corr_math),
        "ols_simple_coef_ratio_enroll_over_teachers": float(
            model_simple.params["ratio_enroll_over_teachers"]
        ),
        "ols_simple_pvalue_ratio_enroll_over_teachers": float(
            model_simple.pvalues["ratio_enroll_over_teachers"]
        ),
        "ols_simple_r2": float(model_simple.rsquared),
        "ols_multi_coef_ratio_enroll_over_teachers": float(
            model_multi.params["ratio_enroll_over_teachers"]
        ),
        "ols_multi_pvalue_ratio_enroll_over_teachers": float(
            model_multi.pvalues["ratio_enroll_over_teachers"]
        ),
        "ols_multi_r2": float(model_multi.rsquared),
    }

    # Print machine-readable JSON so the calling process can inspect it.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
