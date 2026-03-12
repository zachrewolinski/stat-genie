import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the variables of interest
    df = df.dropna(subset=["student_teacher_ratio", "avg_score"])

    n = len(df)
    ratio_mean = df["student_teacher_ratio"].mean()
    ratio_std = df["student_teacher_ratio"].std()
    score_mean = df["avg_score"].mean()
    score_std = df["avg_score"].std()

    corr, corr_p = stats.pearsonr(df["student_teacher_ratio"], df["avg_score"])

    # Simple bivariate regression
    X = sm.add_constant(df["student_teacher_ratio"])
    y = df["avg_score"]
    model = sm.OLS(y, X).fit()
    slope = model.params["student_teacher_ratio"]
    slope_se = model.bse["student_teacher_ratio"]
    slope_t = model.tvalues["student_teacher_ratio"]
    slope_p = model.pvalues["student_teacher_ratio"]
    r2 = model.rsquared

    print("Sample size:", n)
    print(
        f"Student-teacher ratio: mean={ratio_mean:.2f}, std={ratio_std:.2f}, "
        f"min={df['student_teacher_ratio'].min():.2f}, "
        f"max={df['student_teacher_ratio'].max():.2f}",
    )
    print(
        f"Average test score: mean={score_mean:.2f}, std={score_std:.2f}, "
        f"min={df['avg_score'].min():.2f}, "
        f"max={df['avg_score'].max():.2f}",
    )
    print(f"Pearson correlation (ratio, avg_score): {corr:.3f}, p-value={corr_p:.3e}")
    print(
        "OLS regression (bivariate): avg_score = beta0 + beta1 * student_teacher_ratio\n"
        f"  beta1 (slope) = {slope:.3f}\n"
        f"  SE(beta1)     = {slope_se:.3f}\n"
        f"  t-stat(beta1) = {slope_t:.3f}\n"
        f"  p-value(beta1)= {slope_p:.3e}\n"
        f"  R-squared     = {r2:.3f}"
    )

    # Multiple regression with key observed covariates
    covariates = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    df_multi = df.dropna(subset=covariates)
    X_multi = sm.add_constant(df_multi[["student_teacher_ratio"] + covariates])
    y_multi = df_multi["avg_score"]
    model_multi = sm.OLS(y_multi, X_multi).fit()

    slope_m = model_multi.params["student_teacher_ratio"]
    slope_m_se = model_multi.bse["student_teacher_ratio"]
    slope_m_t = model_multi.tvalues["student_teacher_ratio"]
    slope_m_p = model_multi.pvalues["student_teacher_ratio"]
    r2_m = model_multi.rsquared

    print(
        "\nOLS regression (with controls): avg_score = beta0 + beta1 * student_teacher_ratio + controls\n"
        f"  beta1 (slope) = {slope_m:.3f}\n"
        f"  SE(beta1)     = {slope_m_se:.3f}\n"
        f"  t-stat(beta1) = {slope_m_t:.3f}\n"
        f"  p-value(beta1)= {slope_m_p:.3e}\n"
        f"  R-squared     = {r2_m:.3f}"
    )


if __name__ == "__main__":
    main()
