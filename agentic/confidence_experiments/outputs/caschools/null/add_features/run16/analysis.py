import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_data():
    df = pd.read_csv("caschools.csv")
    # Compute student-teacher ratio and average test score
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = df[["read", "math"]].mean(axis=1)
    cols = ["stratio", "avgscore", "calworks", "lunch", "income", "english"]
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return df


def analyze_relationship(df: pd.DataFrame):
    x = df["stratio"]
    y = df["avgscore"]

    # Correlation analysis
    r, p_corr = stats.pearsonr(x, y)

    # Simple linear regression: avgscore ~ stratio
    X1 = sm.add_constant(x)
    model1 = sm.OLS(y, X1).fit()
    coef1 = model1.params["stratio"]
    p1 = model1.pvalues["stratio"]
    ci1_low, ci1_high = model1.conf_int().loc["stratio"]
    r2_1 = model1.rsquared

    # Multiple regression controlling for key demographics
    X2 = df[["stratio", "calworks", "lunch", "income", "english"]]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()
    coef2 = model2.params["stratio"]
    p2 = model2.pvalues["stratio"]
    ci2_low, ci2_high = model2.conf_int().loc["stratio"]
    r2_2 = model2.rsquared

    return {
        "corr_r": float(r),
        "corr_p": float(p_corr),
        "coef1": float(coef1),
        "p1": float(p1),
        "ci1_low": float(ci1_low),
        "ci1_high": float(ci1_high),
        "r2_1": float(r2_1),
        "coef2": float(coef2),
        "p2": float(p2),
        "ci2_low": float(ci2_low),
        "ci2_high": float(ci2_high),
        "r2_2": float(r2_2),
    }


def determine_response(stats_dict):
    r = stats_dict["corr_r"]
    p_corr = stats_dict["corr_p"]
    coef1 = stats_dict["coef1"]
    p1 = stats_dict["p1"]
    coef2 = stats_dict["coef2"]
    p2 = stats_dict["p2"]

    # Default neutral response
    response = 50
    is_yes = False

    # Strong, consistent negative association with significance
    if (
        p_corr < 0.05
        and p1 < 0.05
        and p2 < 0.05
        and r < 0
        and coef1 < 0
        and coef2 < 0
    ):
        if abs(r) < 0.2:
            response = 65
        elif abs(r) < 0.4:
            response = 80
        else:
            response = 90
        is_yes = True
    # Some evidence of a negative association, but weaker or less consistent
    elif (p_corr < 0.05 or p1 < 0.05 or p2 < 0.05) and (r < 0 or coef1 < 0 or coef2 < 0):
        if abs(r) < 0.2:
            response = 55
        elif abs(r) < 0.4:
            response = 70
        else:
            response = 80
        is_yes = True
    else:
        # Little or no convincing evidence
        if abs(r) < 0.1:
            response = 10
        elif abs(r) < 0.2:
            response = 25
        else:
            response = 35
        is_yes = False

    return int(round(response)), is_yes


def build_explanation(stats_dict, response, is_yes):
    direction = "Yes" if is_yes else "No"
    r = stats_dict["corr_r"]
    p_corr = stats_dict["corr_p"]
    coef1 = stats_dict["coef1"]
    ci1_low = stats_dict["ci1_low"]
    ci1_high = stats_dict["ci1_high"]
    r2_1 = stats_dict["r2_1"]
    coef2 = stats_dict["coef2"]
    ci2_low = stats_dict["ci2_low"]
    ci2_high = stats_dict["ci2_high"]
    r2_2 = stats_dict["r2_2"]
    p1 = stats_dict["p1"]
    p2 = stats_dict["p2"]

    explanation = (
        f"{direction}: A lower student–teacher ratio is "
        f"{'associated' if is_yes else 'not clearly associated'} with higher academic performance "
        f"based on this dataset. The Pearson correlation between the student–teacher ratio "
        f"and average test score (mean of reading and math) is r = {r:.3f} (p = {p_corr:.3g}), "
        f"indicating a {'negative' if r < 0 else 'positive' if r > 0 else 'near-zero'} linear association. "
        f"In a simple linear regression of average test score on the student–teacher ratio, "
        f"the slope is {coef1:.3f} points per additional student per teacher "
        f"(95% CI [{ci1_low:.3f}, {ci1_high:.3f}], p = {p1:.3g}, R² = {r2_1:.3f}). "
        f"When controlling for economic disadvantage (CalWorks and lunch eligibility), income, "
        f"and the share of English learners, the estimated effect of the student–teacher ratio remains "
        f"{'negative' if coef2 < 0 else 'positive' if coef2 > 0 else 'near zero'} at {coef2:.3f} "
        f"points per additional student per teacher "
        f"(95% CI [{ci2_low:.3f}, {ci2_high:.3f}], p = {p2:.3g}, R² = {r2_2:.3f}). "
        f"Taken together, these results correspond to a Likert-scale response of {response} out of 100, "
        f"reflecting the strength and statistical reliability of the observed association in this sample."
    )
    return explanation


def main():
    # Load metadata (for context, not strictly needed for computation)
    info_path = Path("info.json")
    if info_path.exists():
        with info_path.open() as f:
            _ = json.load(f)

    df = load_data()
    stats_dict = analyze_relationship(df)
    response, is_yes = determine_response(stats_dict)
    explanation = build_explanation(stats_dict, response, is_yes)

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

