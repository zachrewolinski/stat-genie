import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_column_by_description(fields, substring: str) -> str:
    for field in fields:
        desc = field.get("properties", {}).get("description", "")
        if substring in desc:
            return field["column"]
    raise KeyError(f"No column found with description containing '{substring}'")


def main() -> None:
    info = load_metadata(Path("info.json"))
    fields = info["data_desc"]["fields"]

    # Identify key columns from metadata descriptions
    enrollment_col = get_column_by_description(fields, "Total enrollment")
    teachers_col = get_column_by_description(fields, "Number of teachers")
    read_score_col = get_column_by_description(fields, "Average reading score")
    math_score_col = get_column_by_description(fields, "Average math score")

    calworks_pct_col = get_column_by_description(
        fields, "Percent qualifying for CalWorks"
    )
    lunch_pct_col = get_column_by_description(
        fields, "Percent qualifying for reduced-price lunch"
    )
    english_learner_pct_col = get_column_by_description(
        fields, "Percent of English learners"
    )
    income_col = get_column_by_description(
        fields, "District average income"
    )
    expend_col = get_column_by_description(
        fields, "Expenditure per student"
    )

    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and academic performance measures
    df["str_ratio"] = df[enrollment_col] / df[teachers_col]
    df["read_score"] = df[read_score_col]
    df["math_score"] = df[math_score_col]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    # Basic descriptive statistics
    print("Descriptive statistics:")
    print(df[["str_ratio", "read_score", "math_score", "avg_score"]].describe())
    print()

    # Simple Pearson correlations between STR and scores
    for col in ["read_score", "math_score", "avg_score"]:
        r, p = stats.pearsonr(df["str_ratio"], df[col])
        print(f"Correlation STR vs {col}: r={r:.3f}, p={p:.4g}")
    print()

    # Simple linear regression: academic performance on student-teacher ratio
    y = df["avg_score"]
    X = sm.add_constant(df["str_ratio"])
    ols_simple = sm.OLS(y, X).fit()
    print("Simple OLS: avg_score ~ str_ratio")
    print(ols_simple.summary().as_text())
    print()

    # Multiple regression controlling for key covariates
    covariate_cols = [
        calworks_pct_col,
        lunch_pct_col,
        english_learner_pct_col,
        income_col,
        expend_col,
    ]
    X_multi = sm.add_constant(df[["str_ratio"] + covariate_cols])
    ols_multi = sm.OLS(y, X_multi).fit()
    print("Multiple OLS: avg_score ~ str_ratio + controls")
    print(ols_multi.summary().as_text())


if __name__ == "__main__":
    main()

