import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def run_analysis() -> dict:
    """
    Load the caschools dataset, compute student–teacher ratio and test scores,
    estimate their association, and return a structured conclusion dictionary.
    """
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Core derived variables
    df = df.copy()
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Keep only variables needed for analysis and drop any missing values
    vars_for_model = ["str", "testscr", "income", "english", "lunch", "calworks", "expenditure"]
    df_model = df[vars_for_model].dropna()

    n_obs = len(df_model)

    # Simple correlation between student–teacher ratio and test scores
    corr = df_model["str"].corr(df_model["testscr"])

    # Bivariate OLS: testscr ~ str
    X_simple = sm.add_constant(df_model[["str"]])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    coef_simple = float(model_simple.params["str"])
    pval_simple = float(model_simple.pvalues["str"])

    # Multivariate OLS with key covariates
    covariates = ["income", "english", "lunch", "calworks", "expenditure"]
    X_full = sm.add_constant(df_model[["str"] + covariates])
    model_full = sm.OLS(df_model["testscr"], X_full).fit()
    coef_full = float(model_full.params["str"])
    pval_full = float(model_full.pvalues["str"])
    r2_full = float(model_full.rsquared)

    # Determine whether the evidence supports a negative relationship
    # (higher student–teacher ratio -> lower scores, so lower ratio -> higher scores)
    associated = coef_full < 0 and pval_full < 0.05 and corr < 0
    response = "Yes" if associated else "No"

    # Heuristic confidence score based on consistency and strength of evidence
    if associated:
        if pval_full < 0.001 and abs(corr) > 0.3:
            confidence = 90
        elif pval_full < 0.01 and abs(corr) > 0.2:
            confidence = 80
        else:
            confidence = 70
    else:
        if pval_full > 0.2 or coef_full >= 0:
            confidence = 70
        else:
            confidence = 60

    # Build a human-readable explanation summarizing data and results
    direction_corr = "higher" if corr < 0 else "lower or similar"
    direction_full = "higher" if coef_full < 0 else "similar or lower"
    significance_phrase = "statistically significant " if pval_full < 0.05 else ""

    explanation = (
        f"Using data on {n_obs} California K-6 and K-8 school districts, "
        f"I computed the student–teacher ratio as the number of students divided by the "
        f"number of teachers and defined academic performance as the average of 5th-grade "
        f"reading and math scores. The simple correlation between the student–teacher ratio "
        f"and average test score is {corr:.3f}, indicating that districts with smaller classes "
        f"tend to have {direction_corr} scores. A bivariate ordinary least squares regression "
        f"of average test scores on the student–teacher ratio yields a coefficient of "
        f"{coef_simple:.3f} (p = {pval_simple:.3g}), meaning that an increase of one student "
        f"per teacher is associated with a change of {coef_simple:.2f} points in average "
        f"test scores. After controlling for district characteristics including average income, "
        f"the shares of students on public assistance, eligible for reduced-price lunch, "
        f"and classified as English learners, as well as expenditures per student, the "
        f"coefficient on the student–teacher ratio is {coef_full:.3f} (p = {pval_full:.3g}) "
        f"with an R-squared of {r2_full:.3f}. This adjusted model suggests that, holding these "
        f"factors constant, a lower student–teacher ratio remains {significance_phrase}"
        f"associated with {direction_full} test scores. "
    )

    if associated:
        explanation += (
            "Because the estimated relationship is negative, consistent across correlation and "
            "regression analyses, and statistically significant after adjusting for key covariates, "
            "I conclude that lower student–teacher ratios are associated with higher academic "
            "performance in this dataset. This is an observational association and does not by "
            "itself prove causality, but the pattern is robust within the available data."
        )
    else:
        explanation += (
            "Taken together, these results do not provide strong or consistent evidence that "
            "lower student–teacher ratios are associated with higher academic performance after "
            "accounting for observed district characteristics, so I do not make a strong claim "
            "of association from this dataset alone."
        )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    conclusion = run_analysis()
    # Write the required JSON object to conclusion.txt with no extra content
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

