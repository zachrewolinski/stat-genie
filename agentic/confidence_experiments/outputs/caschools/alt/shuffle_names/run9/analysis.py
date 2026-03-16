import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Map shuffled column names to their semantic meaning based on info.json
    df["enrollment"] = df["english"]  # Total enrollment
    df["teachers_fte"] = df["students"]  # Number of teachers (FTE)
    df["calworks_pct"] = df["school"]  # Percent qualifying for CalWorks
    df["lunch_pct"] = df["computer"]  # Percent qualifying for reduced-price lunch
    df["computers_n"] = df["county"]  # Number of computers
    df["exp_per_student"] = df["grades"]  # Expenditure per student
    df["income_k"] = df["income"]  # District average income (in $1000s)
    df["ell_pct"] = df["rownames"]  # Percent of English learners
    df["read_score"] = df["district"]  # Average reading score
    df["math_score"] = df["expenditure"]  # Average math score

    # Construct student-teacher ratio and overall performance
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["enrollment", "teachers_fte", "read_score", "math_score"])

    df = df[df["teachers_fte"] > 0]
    df["stratio"] = df["enrollment"] / df["teachers_fte"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    # Basic sanity checks
    summary = {
        "n_rows": len(df),
        "stratio_mean": float(df["stratio"].mean()),
        "stratio_std": float(df["stratio"].std()),
        "stratio_min": float(df["stratio"].min()),
        "stratio_max": float(df["stratio"].max()),
    }

    # Simple Pearson correlations
    correlations = {
        "stratio_read_corr": float(df["stratio"].corr(df["read_score"])),
        "stratio_math_corr": float(df["stratio"].corr(df["math_score"])),
        "stratio_avg_corr": float(df["stratio"].corr(df["avg_score"])),
    }

    # Helper to fit OLS and extract slope and p-value for stratio
    def fit_ols(y, x_cols):
        x = sm.add_constant(df[x_cols])
        model = sm.OLS(df[y], x, missing="drop")
        res = model.fit()
        return {
            "coef": {k: float(res.params[k]) for k in res.params.index},
            "pvalues": {k: float(res.pvalues[k]) for k in res.pvalues.index},
            "r2": float(res.rsquared),
        }

    # Unadjusted association: test scores ~ student-teacher ratio
    ols_bivariate = {
        "read": fit_ols("read_score", ["stratio"]),
        "math": fit_ols("math_score", ["stratio"]),
        "avg": fit_ols("avg_score", ["stratio"]),
    }

    # Adjusted association controlling for key socioeconomic and school factors
    covariates = ["stratio", "income_k", "calworks_pct", "lunch_pct", "ell_pct", "exp_per_student"]
    # Some columns may have missing values; drop rows with missing covariates
    df_cov = df.dropna(subset=covariates).copy()

    def fit_ols_cov(y):
        x = sm.add_constant(df_cov[covariates])
        model = sm.OLS(df_cov[y], x, missing="drop")
        res = model.fit()
        return {
            "coef": {k: float(res.params[k]) for k in res.params.index},
            "pvalues": {k: float(res.pvalues[k]) for k in res.pvalues.index},
            "r2": float(res.rsquared),
            "n": int(res.nobs),
        }

    ols_adjusted = {
        "read": fit_ols_cov("read_score"),
        "math": fit_ols_cov("math_score"),
        "avg": fit_ols_cov("avg_score"),
    }

    results = {
        "summary": summary,
        "correlations": correlations,
        "ols_bivariate": ols_bivariate,
        "ols_adjusted": ols_adjusted,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

