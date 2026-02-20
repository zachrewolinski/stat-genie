import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


DATA_FILE = Path("caschools.csv")
CONCLUSION_FILE = Path("conclusion.txt")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Student–teacher ratio: number of students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Average academic performance across reading and math
    df["avgscore"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_correlation(df: pd.DataFrame):
    r, p = stats.pearsonr(df["stratio"], df["avgscore"])
    return float(r), float(p)


def regression_analysis(df: pd.DataFrame):
    """
    Run an OLS regression of average score on student–teacher ratio
    and key demographic controls.
    """
    predictors = ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    X = df[predictors].copy()
    X = sm.add_constant(X)
    y = df["avgscore"]

    model = sm.OLS(y, X).fit()
    coef = float(model.params["stratio"])
    p_value = float(model.pvalues["stratio"])
    r_squared = float(model.rsquared)
    return {
        "coef_stratio": coef,
        "p_value_stratio": p_value,
        "r_squared": r_squared,
    }


def quartile_difference(df: pd.DataFrame):
    """
    Compare mean scores between districts in the lowest and highest
    quartiles of student–teacher ratio.
    """
    df = df.copy()
    df["str_q"] = pd.qcut(df["stratio"], 4, labels=False)
    low = df[df["str_q"] == 0]["avgscore"]
    high = df[df["str_q"] == 3]["avgscore"]

    diff = float(low.mean() - high.mean())
    # t-test for difference in means (Welch's t-test)
    t_stat, p_val = stats.ttest_ind(low, high, equal_var=False)
    return {
        "mean_low": float(low.mean()),
        "mean_high": float(high.mean()),
        "diff_low_minus_high": diff,
        "p_value_diff": float(p_val),
    }


def derive_conclusion(corr_res, reg_res, q_res):
    """
    Turn numerical results into the required structured conclusion.
    """
    r, p_corr = corr_res
    coef = reg_res["coef_stratio"]
    p_reg = reg_res["p_value_stratio"]
    r2 = reg_res["r_squared"]
    diff = q_res["diff_low_minus_high"]
    p_diff = q_res["p_value_diff"]

    # Interpretation:
    # Lower student–teacher ratio corresponds to lower "stratio".
    # If higher scores are observed when stratio is lower, then
    # the association is that lower student–teacher ratio is
    # associated with higher academic performance.
    #
    # This means:
    # - Negative correlation between stratio and avgscore
    # - Negative regression coefficient on stratio
    # - Positive difference (low-ratio minus high-ratio) in mean scores
    indicators_support_yes = 0
    indicators_support_no = 0

    # Correlation indicator
    if p_corr < 0.05:
        if r < 0:
            indicators_support_yes += 1
        else:
            indicators_support_no += 1

    # Regression indicator
    if p_reg < 0.05:
        if coef < 0:
            indicators_support_yes += 1
        else:
            indicators_support_no += 1

    # Quartile difference indicator
    if p_diff < 0.05:
        if diff > 0:
            indicators_support_yes += 1
        else:
            indicators_support_no += 1

    if indicators_support_yes > indicators_support_no:
        response = "Yes"
    elif indicators_support_no > indicators_support_yes:
        response = "No"
    else:
        # If evidence is mixed or very weak, answer based on sign of
        # regression coefficient as primary measure.
        response = "Yes" if coef < 0 else "No"

    # Strength: heuristic based on consistency and magnitude
    strength = 50

    # Use standardized effect from correlation as a base indicator
    abs_r = abs(r)
    if abs_r >= 0.4:
        strength += 15
    elif abs_r >= 0.2:
        strength += 5

    if p_corr < 0.01:
        strength += 10
    elif p_corr < 0.05:
        strength += 5

    if p_reg < 0.01:
        strength += 10
    elif p_reg < 0.05:
        strength += 5

    if indicators_support_yes == 3 or indicators_support_no == 3:
        strength += 10

    strength = max(0, min(100, strength))

    # Confidence: reflects data size and model robustness rather than
    # direction alone.
    confidence = 60
    # Larger sample and decent R^2 increase confidence
    if r2 >= 0.2:
        confidence += 15
    elif r2 >= 0.1:
        confidence += 5

    if p_corr < 0.01 and p_reg < 0.01:
        confidence += 15
    elif p_corr < 0.05 and p_reg < 0.05:
        confidence += 5

    confidence = max(0, min(100, confidence))

    explanation = {
        "summary": (
            "I examined the association between student–teacher ratio "
            "and average academic performance (mean of reading and math scores) "
            "across 420 California K-6 and K-8 districts."
        ),
        "findings": {
            "correlation": {
                "r_stratio_avgscore": r,
                "p_value": p_corr,
            },
            "regression": reg_res,
            "quartile_comparison": q_res,
        },
        "interpretation": (
            "Lower student–teacher ratios correspond to lower values of the "
            "students-per-teacher measure. Negative correlations and negative "
            "regression coefficients therefore imply that districts with fewer "
            "students per teacher tend to have higher average test scores, "
            "after accounting for income, English-learner share, poverty "
            "proxies, and spending. The quartile comparison compares mean "
            "scores between districts with the smallest and largest ratios."
        ),
        "limitations": (
            "The analysis is observational and cross-sectional; it cannot prove "
            "causality, and unmeasured district characteristics may confound the "
            "relationship. The linear specification may miss non-linear effects."
        ),
    }

    return response, strength, confidence, explanation


def main():
    df = load_data()
    corr_res = simple_correlation(df)
    reg_res = regression_analysis(df)
    q_res = quartile_difference(df)

    response, strength, confidence, explanation = derive_conclusion(
        corr_res, reg_res, q_res
    )

    # Log key numeric results for transparency when running the script
    print("Correlation (stratio vs avgscore):", corr_res)
    print("Regression results:", reg_res)
    print("Quartile comparison:", q_res)
    print("Derived response:", response)
    print("Strength:", strength)
    print("Confidence:", confidence)

    conclusion_obj = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    CONCLUSION_FILE.write_text(json.dumps(conclusion_obj, indent=None))


if __name__ == "__main__":
    main()

