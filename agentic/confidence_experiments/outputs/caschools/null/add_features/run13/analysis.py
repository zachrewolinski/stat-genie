import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in the variables of interest
    sub = df[["stratio", "testscr"]].dropna()

    # Correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(sub["stratio"], sub["testscr"])

    # Simple OLS regression: testscr ~ stratio
    X = sm.add_constant(sub["stratio"])
    model = sm.OLS(sub["testscr"], X).fit()
    coef_stratio = float(model.params["stratio"])
    p_coef = float(model.pvalues["stratio"])
    r_squared = float(model.rsquared)

    # Map evidence to a 0-100 Likert-style response score
    # 0 = strong "No", 100 = strong "Yes" to:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    #
    # A negative association (higher ratio -> lower scores) supports the statement.
    if p_corr >= 0.05 or p_coef >= 0.05:
        # Little evidence of a relationship
        response = 50
        conclusion_sentence = (
            "Because the association is very small in magnitude and not statistically "
            "significant, this dataset does not provide strong evidence that lower "
            "student-teacher ratios are associated with higher academic performance."
        )
    else:
        # Significant relationship; scale strength by effect size (|r|)
        abs_r = abs(r)
        if r < 0:
            # Evidence consistent with the research hypothesis
            if abs_r < 0.1:
                response = 70
            elif abs_r < 0.3:
                response = 80
            else:
                response = 90
            conclusion_sentence = (
                "Because the association is negative and statistically significant, the "
                "data support the view that lower student-teacher ratios are associated "
                "with higher academic performance."
            )
        else:
            # Evidence in the opposite direction
            if abs_r < 0.1:
                response = 30
            elif abs_r < 0.3:
                response = 20
            else:
                response = 10
            conclusion_sentence = (
                "Because the association is positive and statistically significant, the "
                "data do not support the claim that lower student-teacher ratios are "
                "associated with higher academic performance in this dataset."
            )

    response = int(np.clip(round(response), 0, 100))

    # Describe direction and strength of correlation
    if abs(r) < 0.05:
        strength_desc = "essentially zero (no clear linear relationship)"
    elif abs(r) < 0.1:
        strength_desc = "very weak"
    elif abs(r) < 0.3:
        strength_desc = "modest"
    else:
        strength_desc = "moderate to strong"

    if r < 0:
        direction_desc = (
            "negative, meaning that higher student-teacher ratios tend to be associated "
            "with lower test scores"
        )
    elif r > 0:
        direction_desc = (
            "positive, meaning that higher student-teacher ratios tend to be associated "
            "with higher test scores"
        )
    else:
        direction_desc = "exactly zero"

    # Build explanation string with key numerical evidence
    explanation = (
        "Using data from 420 California K-6/K-8 districts, "
        "I constructed the student-teacher ratio as students divided by teachers "
        "and an academic performance measure as the average of 5th-grade reading "
        "and math scores. "
        f"The Pearson correlation between student-teacher ratio and average test score "
        f"is {r:.3f} (p = {p_corr:.2e}); in magnitude this is {strength_desc} and the "
        f"relationship is {direction_desc}. "
        f"A linear regression of average test score on student-teacher ratio yields a "
        f"coefficient of {coef_stratio:.3f} points per one-student increase in the ratio "
        f"(p = {p_coef:.2e}, R-squared = {r_squared:.3f}). "
        + conclusion_sentence
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
