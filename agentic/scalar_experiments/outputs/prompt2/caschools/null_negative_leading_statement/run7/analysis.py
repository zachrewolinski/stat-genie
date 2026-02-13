import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    # Paths
    data_path = Path("caschools.csv")
    info_path = Path("info.json")
    conclusion_path = Path("conclusion.txt")

    # Load data and metadata
    df = pd.read_csv(data_path)
    with info_path.open("r") as f:
        info = json.load(f)

    # Construct key variables
    # Student-teacher ratio (higher = more students per teacher)
    df["str"] = df["students"] / df["teachers"]
    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (should be none, but be safe)
    df_model = df.dropna(subset=["str", "testscr", "income", "english", "lunch", "calworks"])

    # Simple bivariate association
    X_simple = sm.add_constant(df_model[["str"]])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression with key demographic controls
    controls = ["income", "english", "lunch", "calworks"]
    X_full = sm.add_constant(df_model[["str"] + controls])
    model_full = sm.OLS(y, X_full).fit()

    # Extract key statistics for student-teacher ratio
    coef_simple = model_simple.params["str"]
    pval_simple = model_simple.pvalues["str"]

    coef_full = model_full.params["str"]
    pval_full = model_full.pvalues["str"]

    # Also look at descriptive relationship between quartiles of STR and test scores
    df_model["str_quartile"] = pd.qcut(df_model["str"], 4, labels=False)
    mean_by_quartile = df_model.groupby("str_quartile")["testscr"].mean()

    # Determine answer: Is lower STR associated with higher performance?
    # We look for a negative and reasonably stable coefficient on STR
    evidence_negative = (coef_simple < 0) and (coef_full < 0)
    evidence_significant = (pval_simple < 0.05) and (pval_full < 0.1)

    # Also check whether mean testscr in lowest-STR quartile exceeds highest-STR quartile
    descriptive_support = bool(mean_by_quartile.iloc[0] > mean_by_quartile.iloc[-1])

    if evidence_negative and (evidence_significant or descriptive_support):
        response = "Yes"
    else:
        response = "No"

    # Set confidence based on consistency of evidence
    if response == "Yes":
        if evidence_negative and evidence_significant and descriptive_support:
            confidence = 85
        elif evidence_negative and (evidence_significant or descriptive_support):
            confidence = 70
        else:
            confidence = 55
    else:
        if not evidence_negative and not descriptive_support:
            confidence = 80
        else:
            confidence = 60

    # Build explanation string with key numerical evidence
    explanation_lines = []
    explanation_lines.append(
        "We constructed a student–teacher ratio (number of students per teacher) "
        "and an overall test score equal to the average of reading and math scores for each district."
    )
    explanation_lines.append(
        f"In a simple OLS regression of average test score on the student–teacher ratio, "
        f"the estimated coefficient on the ratio was {coef_simple:.3f} with p-value {pval_simple:.3f}."
    )
    explanation_lines.append(
        f"In a multiple regression controlling for district income, the percentage of English learners, "
        f"and poverty indicators (CalWorks and reduced-price lunch), the coefficient on the ratio was "
        f"{coef_full:.3f} with p-value {pval_full:.3f}."
    )
    explanation_lines.append(
        f"Descriptively, districts in the lowest student–teacher-ratio quartile had an average test score of "
        f"{mean_by_quartile.iloc[0]:.1f}, compared with {mean_by_quartile.iloc[-1]:.1f} in the highest-ratio quartile."
    )
    explanation_lines.append(
        "Combining the regression and descriptive evidence, we "
        + ("find" if response == "Yes" else "do not find")
        + " clear support for the claim that lower student–teacher ratios are associated with higher academic performance in this dataset."
    )

    explanation = " " .join(explanation_lines)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with conclusion_path.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
