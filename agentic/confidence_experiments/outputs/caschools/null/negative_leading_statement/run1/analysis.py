import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).parent

    info_path = base / "info.json"
    data_path = base / "caschools.csv"

    info = json.loads(info_path.read_text())
    question = info["research_questions"][0]

    df = pd.read_csv(data_path)

    # Construct student-teacher ratio (students per teacher).
    df = df.copy()
    df["str"] = df["students"] / df["teachers"]

    # Academic performance: analyze both reading and math, plus their average.
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    results = {}

    # Simple bivariate regressions: score ~ str
    for outcome in ["read", "math", "avg_score"]:
        y = df[outcome]
        X = sm.add_constant(df["str"])
        model = sm.OLS(y, X, missing="drop").fit()
        coef = model.params["str"]
        pval = model.pvalues["str"]
        r2 = model.rsquared
        results[f"{outcome}_bivariate"] = {
            "coef": float(coef),
            "pval": float(pval),
            "r2": float(r2),
        }

    # Multiple regression with key observed confounders.
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer", "students"]
    for outcome in ["read", "math", "avg_score"]:
        y = df[outcome]
        X = df[["str"] + controls]
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing="drop").fit()
        coef = model.params["str"]
        pval = model.pvalues["str"]
        r2 = model.rsquared
        results[f"{outcome}_adjusted"] = {
            "coef": float(coef),
            "pval": float(pval),
            "r2": float(r2),
        }

    # Also look at correlation.
    corr_read = df["str"].corr(df["read"])
    corr_math = df["str"].corr(df["math"])
    corr_avg = df["str"].corr(df["avg_score"])

    summary = {
        "question": question,
        "n": int(df.shape[0]),
        "results": results,
        "correlations": {
            "read": float(corr_read),
            "math": float(corr_math),
            "avg_score": float(corr_avg),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
