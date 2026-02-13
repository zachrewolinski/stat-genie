import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Student-teacher ratio: students per teacher.
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores.
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables (if any).
    sub = df[["stratio", "testscr"]].dropna()

    # Simple diagnostics: correlation and OLS regression.
    corr = sub["stratio"].corr(sub["testscr"])

    X = sm.add_constant(sub["stratio"])
    model = sm.OLS(sub["testscr"], X).fit()
    slope = model.params["stratio"]
    p_value = model.pvalues["stratio"]

    # Decide on answer based on sign and statistical significance.
    # "Yes" if lower student-teacher ratio (fewer students per teacher)
    # is associated with higher test scores, i.e., a statistically
    # significant negative slope.
    alpha = 0.05
    if slope < 0 and p_value < alpha:
        response = "Yes"
    else:
        response = "No"

    base_explanation = (
        "Using 420 California school districts, I computed the student-teacher "
        "ratio as students per teacher and an academic performance score as the "
        "average of 5th-grade reading and math test scores. A simple OLS "
        "regression of test scores on the student-teacher ratio shows a "
        f"slope of {slope:.3f} with p-value {p_value:.4f}, and the Pearson "
        f"correlation between ratio and scores is {corr:.3f}. "
    )

    if response == "Yes":
        tail = (
            "This indicates that lower student-teacher ratios (fewer students per "
            "teacher) are associated with higher academic performance in this "
            "dataset."
        )
    else:
        tail = (
            "In this analysis, the estimated association is not both negative and "
            "statistically significant at the 5% level, so we do not find strong "
            "evidence that lower student-teacher ratios are associated with higher "
            "academic performance in this dataset."
        )

    explanation = base_explanation + tail

    conclusion = {"response": response, "explanation": explanation}
    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
