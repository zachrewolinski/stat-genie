import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    analysis_df = df[["student_teacher_ratio", "avg_score"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    x = analysis_df["student_teacher_ratio"]
    y = analysis_df["avg_score"]

    # Correlation analysis
    corr, corr_p = pearsonr(x, y)

    # Simple linear regression: avg_score ~ student_teacher_ratio
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = float(model.params["student_teacher_ratio"])
    slope_p = float(model.pvalues["student_teacher_ratio"])
    r2 = float(model.rsquared)

    # Decide binary answer: lower ratio associated with higher performance?
    # This corresponds to a negative and statistically significant slope.
    alpha = 0.05
    associated = slope < 0 and slope_p < alpha

    if associated:
        response = "Yes"
        interpretation = (
            "Districts with fewer students per teacher tend to have higher "
            "average test scores."
        )
    else:
        response = "No"
        interpretation = (
            "The data do not show a clear or statistically significant link "
            "between student-teacher ratio and average test scores."
        )

    explanation = (
        "Using data on 420 K-6 and K-8 California school districts, I computed "
        "student-teacher ratio as total enrollment divided by number of teachers "
        "and academic performance as the average of reading and math scores. "
        f"The Pearson correlation between student-teacher ratio and average score "
        f"is {corr:.3f} (p = {corr_p:.3g}), and a linear regression of average "
        f"score on student-teacher ratio yields a slope of {slope:.2f} points per "
        f"additional student per teacher (p = {slope_p:.3g}, R² = {r2:.3f}). "
        f"{interpretation} This is an observational association and should not be "
        "interpreted as a causal effect."
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

