import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("caschools.csv")

    # Map relevant fields
    students = df["feature6"]
    teachers = df["feature7"]
    read_score = df["feature14"]
    math_score = df["feature15"]

    # Student-teacher ratio and academic performance
    df["str"] = students / teachers
    df["avg_score"] = (read_score + math_score) / 2.0

    # Basic association metrics
    corr = df[["str", "avg_score"]].corr().loc["str", "avg_score"]

    # OLS regression: avg_score ~ str
    X = sm.add_constant(df["str"])
    model = sm.OLS(df["avg_score"], X).fit()

    print("Correlation (STR vs Avg Score):", corr)
    print(model.summary())

    # Save a small results file for reference if needed
    results = {
        "corr": float(corr),
        "coef_str": float(model.params["str"]),
        "pvalue_str": float(model.pvalues["str"]),
        "r2": float(model.rsquared),
    }
    pd.Series(results).to_csv("analysis_results.csv")


if __name__ == "__main__":
    main()
