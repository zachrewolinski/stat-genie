import json
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def load_data() -> pd.DataFrame:
    """Load the caschools dataset and construct key derived variables."""
    df = pd.read_csv("caschools.csv")

    # Rename a subset of columns for readability
    df = df.rename(
        columns={
            "feature6": "enrollment",
            "feature7": "teachers",
            "feature8": "calworks_pct",
            "feature9": "lunch_pct",
            "feature10": "computers",
            "feature11": "expenditure_per_student",
            "feature12": "avg_income_k",
            "feature13": "english_learner_pct",
            "feature14": "reading_score",
            "feature15": "math_score",
        }
    )

    # Derived measures
    df["stratio"] = df["enrollment"] / df["teachers"]
    df["testscr"] = (df["reading_score"] + df["math_score"]) / 2.0
    df["log_enrollment"] = np.log(df["enrollment"])
    df["computers_per_student"] = df["computers"] / df["enrollment"]

    # Drop any rows with invalid ratios or missing key variables
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "stratio",
            "testscr",
            "calworks_pct",
            "lunch_pct",
            "english_learner_pct",
            "log_enrollment",
            "expenditure_per_student",
            "avg_income_k",
            "computers_per_student",
        ]
    )
    return df


def run_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Run correlation and regression analyses linking student-teacher ratio to achievement."""
    # Basic descriptives
    stratio = df["stratio"]
    testscr = df["testscr"]

    stratio_desc = stratio.describe()
    testscr_desc = testscr.describe()

    # Simple correlation
    r, p_corr = stats.pearsonr(stratio, testscr)

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(stratio)
    model_simple = sm.OLS(testscr, X_simple).fit()

    coef_stratio = float(model_simple.params["stratio"])
    se_stratio = float(model_simple.bse["stratio"])
    p_stratio = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key covariates for robustness
    covariates = df[
        [
            "stratio",
            "calworks_pct",
            "lunch_pct",
            "english_learner_pct",
            "log_enrollment",
            "expenditure_per_student",
            "avg_income_k",
            "computers_per_student",
        ]
    ]
    X_controls = sm.add_constant(covariates)
    model_controls = sm.OLS(testscr, X_controls).fit()

    coef_stratio_adj = float(model_controls.params["stratio"])
    se_stratio_adj = float(model_controls.bse["stratio"])
    p_stratio_adj = float(model_controls.pvalues["stratio"])
    r2_controls = float(model_controls.rsquared)

    # Check for simple non-linearity: add squared term
    df = df.copy()
    df["stratio_sq"] = df["stratio"] ** 2
    X_quad = sm.add_constant(df[["stratio", "stratio_sq"]])
    model_quad = sm.OLS(df["testscr"], X_quad).fit()

    coef_stratio_quad = float(model_quad.params["stratio"])
    coef_stratio_sq = float(model_quad.params["stratio_sq"])
    p_stratio_sq = float(model_quad.pvalues["stratio_sq"])

    return {
        "n": int(len(df)),
        "stratio_mean": float(stratio_desc["mean"]),
        "stratio_std": float(stratio_desc["std"]),
        "stratio_min": float(stratio_desc["min"]),
        "stratio_max": float(stratio_desc["max"]),
        "testscr_mean": float(testscr_desc["mean"]),
        "testscr_std": float(testscr_desc["std"]),
        "testscr_min": float(testscr_desc["min"]),
        "testscr_max": float(testscr_desc["max"]),
        "corr_r": float(r),
        "corr_p": float(p_corr),
        "coef_stratio": coef_stratio,
        "se_stratio": se_stratio,
        "p_stratio": p_stratio,
        "r2_simple": r2_simple,
        "coef_stratio_adj": coef_stratio_adj,
        "se_stratio_adj": se_stratio_adj,
        "p_stratio_adj": p_stratio_adj,
        "r2_controls": r2_controls,
        "coef_stratio_quad": coef_stratio_quad,
        "coef_stratio_sq": coef_stratio_sq,
        "p_stratio_sq": p_stratio_sq,
    }


def determine_response(stats_dict: Dict[str, Any]) -> int:
    """Map statistical evidence into a 0–100 Likert-style confidence score."""
    coef = stats_dict["coef_stratio"]
    coef_adj = stats_dict["coef_stratio_adj"]
    p = stats_dict["p_stratio"]
    p_adj = stats_dict["p_stratio_adj"]
    r = stats_dict["corr_r"]
    r2_simple = stats_dict["r2_simple"]
    r2_controls = stats_dict["r2_controls"]

    # Default: ambivalent
    score = 50

    # Evidence for a negative association (lower ratio → higher scores)
    if coef < 0 and coef_adj < 0:
        # Very strong, consistent association with good fit
        if p < 1e-8 and p_adj < 1e-6 and abs(r) >= 0.3 and r2_simple >= 0.1:
            score = 90
        # Strong but slightly weaker evidence
        elif p < 1e-4 and p_adj < 1e-3:
            score = 80
        # Moderately strong evidence
        elif p < 0.01 and p_adj < 0.05:
            score = 70
        else:
            score = 60
    else:
        # Coefficients not consistently negative or not significant
        if p > 0.1 and p_adj > 0.1 and abs(r) < 0.1:
            score = 20
        elif p > 0.05 and p_adj > 0.05:
            score = 40

    # Ensure score is an integer between 0 and 100
    score = int(round(min(max(score, 0), 100)))
    return score


def build_explanation(stats_dict: Dict[str, Any], response: int) -> str:
    """Create a concise natural-language explanation summarizing the evidence."""
    n = stats_dict["n"]
    str_mean = stats_dict["stratio_mean"]
    str_std = stats_dict["stratio_std"]
    str_min = stats_dict["stratio_min"]
    str_max = stats_dict["stratio_max"]
    ts_mean = stats_dict["testscr_mean"]
    ts_std = stats_dict["testscr_std"]
    r = stats_dict["corr_r"]
    p_corr = stats_dict["corr_p"]
    coef = stats_dict["coef_stratio"]
    se = stats_dict["se_stratio"]
    p = stats_dict["p_stratio"]
    r2_simple = stats_dict["r2_simple"]
    coef_adj = stats_dict["coef_stratio_adj"]
    se_adj = stats_dict["se_stratio_adj"]
    p_adj = stats_dict["p_stratio_adj"]
    r2_controls = stats_dict["r2_controls"]
    coef_sq = stats_dict["coef_stratio_sq"]
    p_sq = stats_dict["p_stratio_sq"]

    # Interpretation of the simple correlation
    if r < -0.05:
        corr_phrase = (
            "indicating that districts with more students per teacher tend to have lower test scores."
        )
    elif r > 0.05:
        corr_phrase = (
            "indicating that districts with more students per teacher tend to have slightly higher test scores."
        )
    else:
        corr_phrase = (
            "indicating little to no systematic relationship between the student–teacher ratio and test scores."
        )

    # Interpretation of the linear regression coefficient
    if coef < -0.05:
        direction_phrase = (
            "each additional student per teacher was associated with lower average test scores, "
            "consistent with the idea that smaller classes are linked to better performance"
        )
    elif coef > 0.05:
        direction_phrase = (
            "each additional student per teacher was associated with slightly higher average test scores, "
            "which goes against the simple expectation that smaller classes perform better"
        )
    else:
        direction_phrase = (
            "the estimated effect of the student–teacher ratio on test scores was extremely small in practical terms"
        )

    # Interpretation of non-linearity
    if p_sq >= 0.05:
        nonlinearity_phrase = (
            "The squared term for the student–teacher ratio was small and not strongly significant in a model that "
            "included both the linear and squared terms, suggesting that within the observed range a simple roughly "
            "linear relationship is adequate."
        )
    else:
        nonlinearity_phrase = (
            "The squared term for the student–teacher ratio showed some evidence of non-linearity, so the association "
            "may strengthen or weaken at the extremes of the observed ratios."
        )

    # Overall assessment phrase based on the response score
    if response >= 80:
        overall_phrase = (
            "Overall, the statistical evidence from correlations and regression models points to a clear and practically "
            "meaningful association between lower student–teacher ratios and higher academic performance at the district "
            "level in this dataset, although these are observational data and cannot on their own prove a causal effect."
        )
    elif response >= 60:
        overall_phrase = (
            "Overall, the statistical evidence suggests a modest association between lower student–teacher ratios and higher "
            "academic performance, but the magnitude is moderate and the results could still be influenced by unmeasured "
            "differences across districts."
        )
    elif response > 40:
        overall_phrase = (
            "Overall, the evidence is mixed: some model specifications hint at a possible association between lower "
            "student–teacher ratios and higher academic performance, but the estimates are unstable and surrounded by "
            "substantial uncertainty."
        )
    else:
        overall_phrase = (
            "Overall, the evidence provides little support for a meaningful association between lower student–teacher ratios "
            "and higher academic performance in this dataset; any relationship that exists is likely small relative to other "
            "district characteristics, and the analyses here cannot demonstrate a causal effect."
        )

    explanation = (
        f"I analyzed data on {n} California K-6 and K-8 school districts, focusing on whether a lower student–teacher ratio "
        f"is associated with higher academic performance. I defined the student–teacher ratio as total enrollment divided by "
        f"the number of teachers in each district and summarized it with a mean of {str_mean:.1f} students per teacher "
        f"(standard deviation {str_std:.1f}, range {str_min:.1f} to {str_max:.1f}). Academic performance was measured as the "
        f"average of the district-level 5th-grade reading and math scores, which had a mean of {ts_mean:.1f} points "
        f"(standard deviation {ts_std:.1f}). The Pearson correlation between the student–teacher ratio and average test score "
        f"was r = {r:.3f} with p-value {p_corr:.2e}, {corr_phrase} In a simple linear regression of test scores on the "
        f"student–teacher ratio, each additional student per teacher was associated with a change of {coef:.2f} points in the "
        f"average test score (standard error {se:.2f}, p-value {p:.2e}, R-squared {r2_simple:.3f}); {direction_phrase}. To "
        f"check robustness, I estimated a "
        f"multiple regression that adjusted for student demographics (percent CalWorks, reduced-price lunch, and English learners), "
        f"district size, expenditure per student, average district income, and computers per student; in this model, the estimated "
        f"effect of the student–teacher ratio remained at {coef_adj:.2f} points per additional student per teacher "
        f"(standard error {se_adj:.2f}, p-value {p_adj:.2e}, R-squared {r2_controls:.3f}), indicating how much of the remaining "
        f"variation in test scores is explained after accounting for these other observed characteristics. {nonlinearity_phrase} "
        f"{overall_phrase} Based on this evidence, my overall confidence that a lower student–teacher ratio is associated with "
        f"higher academic performance corresponds to a score of {response} on a 0 to 100 scale, where higher values indicate "
        f"stronger support for a positive association."
    )

    return explanation


def main() -> None:
    df = load_data()
    stats_dict = run_analysis(df)
    response = determine_response(stats_dict)
    explanation = build_explanation(stats_dict, response)

    result = {"response": int(response), "explanation": explanation}

    # Write JSON object with no extra lines or text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
