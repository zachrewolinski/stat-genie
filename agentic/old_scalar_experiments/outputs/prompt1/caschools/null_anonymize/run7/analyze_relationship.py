import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_str(df: pd.DataFrame) -> pd.Series:
    return df["feature6"] / df["feature7"]


def compute_testscr(df: pd.DataFrame) -> pd.Series:
    return (df["feature14"] + df["feature15"]) / 2


def ols_summary(y: pd.Series, x: pd.Series) -> dict:
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    coef = model.params["feature"]
    se = model.bse["feature"]
    pval = model.pvalues["feature"]
    r2 = model.rsquared
    return {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "r2": float(r2),
    }


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df["student_teacher_ratio"] = compute_str(df)
    df["testscr"] = compute_testscr(df)

    # Basic correlation
    corr = df[["student_teacher_ratio", "testscr"]].corr().iloc[0, 1]

    # Simple OLS on the full data
    y = df["testscr"]
    x_full = df["student_teacher_ratio"].rename("feature")
    ols_full = ols_summary(y, x_full)

    # Trim obviously implausible ratios (e.g., <5 or >40 students per teacher)
    trimmed = df[(df["student_teacher_ratio"] >= 5) & (df["student_teacher_ratio"] <= 40)].copy()
    corr_trim = trimmed[["student_teacher_ratio", "testscr"]].corr().iloc[0, 1]
    x_trim = trimmed["student_teacher_ratio"].rename("feature")
    y_trim = trimmed["testscr"]
    ols_trim = ols_summary(y_trim, x_trim)

    # Multiple regression with key covariates on trimmed data
    covariates = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X_cov = sm.add_constant(trimmed[["student_teacher_ratio"] + covariates])
    model_cov = sm.OLS(y_trim, X_cov).fit()
    coef_cov = model_cov.params["student_teacher_ratio"]
    se_cov = model_cov.bse["student_teacher_ratio"]
    pval_cov = model_cov.pvalues["student_teacher_ratio"]

    results = {
        "n_full": int(len(df)),
        "n_trim": int(len(trimmed)),
        "corr_full": float(corr),
        "corr_trim": float(corr_trim),
        "ols_full": ols_full,
        "ols_trim": ols_trim,
        "covariate_model": {
            "coef": float(coef_cov),
            "se": float(se_cov),
            "pval": float(pval_cov),
            "r2": float(model_cov.rsquared),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

