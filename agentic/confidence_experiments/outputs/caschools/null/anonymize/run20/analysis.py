import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).parent
    data_path = base / "caschools.csv"
    info_path = base / "info.json"

    with info_path.open() as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables based on metadata descriptions.
    # Student-teacher ratio ≈ total enrollment / number of teachers.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Center the ratio for easier interpretation.
    df["stratio_c"] = df["student_teacher_ratio"] - df["student_teacher_ratio"].mean()

    print("Research question:", question)
    print("\nBasic descriptions:")
    print(df[["student_teacher_ratio", "avg_score"]].describe())

    # Simple Pearson correlation
    corr = df["student_teacher_ratio"].corr(df["avg_score"])
    print("\nPearson correlation between student-teacher ratio and average score:", corr)

    # Bivariate regression: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    print("\nBivariate OLS: avg_score ~ student_teacher_ratio")
    print(model_simple.summary())

    # Multivariate regression controlling for key socioeconomic covariates
    covariates = [
        "stratio_c",  # centered ratio
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]

    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["avg_score"], X_multi).fit(cov_type="HC3")

    print("\nMultivariate OLS with controls (robust SE, HC3):")
    print(model_multi.summary())

    # Extract key statistics for convenience
    coef_stratio = model_multi.params["stratio_c"]
    se_stratio = model_multi.bse["stratio_c"]
    t_stratio = model_multi.tvalues["stratio_c"]
    p_stratio = model_multi.pvalues["stratio_c"]

    print("\nKey coefficient for centered student-teacher ratio (multivariate model):")
    print(f"coef={coef_stratio:.4f}, se={se_stratio:.4f}, t={t_stratio:.2f}, p={p_stratio:.4g}")

    # Also show the R-squared of simple vs multivariate model
    print(f"\nR-squared (bivariate): {model_simple.rsquared:.3f}")
    print(f"R-squared (multivariate): {model_multi.rsquared:.3f}")

    # Simple robustness check: trim extreme student-teacher ratios (top 5%)
    q95 = df["student_teacher_ratio"].quantile(0.95)
    df_trim = df[df["student_teacher_ratio"] <= q95].copy()
    df_trim["stratio_c"] = df_trim["student_teacher_ratio"] - df_trim["student_teacher_ratio"].mean()

    print("\n=== Trimmed sample analysis (student_teacher_ratio <= 95th percentile) ===")
    print(df_trim[["student_teacher_ratio", "avg_score"]].describe())

    X_simple_trim = sm.add_constant(df_trim["student_teacher_ratio"])
    model_simple_trim = sm.OLS(df_trim["avg_score"], X_simple_trim).fit()
    print("\nBivariate OLS (trimmed): avg_score ~ student_teacher_ratio")
    print(model_simple_trim.summary())

    X_multi_trim = sm.add_constant(
        df_trim[
            [
                "stratio_c",
                "feature8",
                "feature9",
                "feature11",
                "feature12",
                "feature13",
            ]
        ]
    )
    model_multi_trim = sm.OLS(df_trim["avg_score"], X_multi_trim).fit(cov_type="HC3")
    print("\nMultivariate OLS with controls (trimmed, robust SE, HC3):")
    print(model_multi_trim.summary())

    coef_stratio_trim = model_multi_trim.params["stratio_c"]
    se_stratio_trim = model_multi_trim.bse["stratio_c"]
    t_stratio_trim = model_multi_trim.tvalues["stratio_c"]
    p_stratio_trim = model_multi_trim.pvalues["stratio_c"]

    print("\nKey coefficient for centered student-teacher ratio (trimmed multivariate model):")
    print(
        f"coef={coef_stratio_trim:.4f}, se={se_stratio_trim:.4f}, "
        f"t={t_stratio_trim:.2f}, p={p_stratio_trim:.4g}"
    )



if __name__ == "__main__":
    main()
