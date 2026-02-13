import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def format_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    df = df.copy()
    # Student–teacher ratio: enrollment / teachers
    df["stratio"] = df["feature6"] / df["feature7"]
    # Overall test score: average of reading and math scores
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    df = df.dropna(subset=["stratio", "testscr"])

    # Basic descriptive statistics
    mean_stratio = float(df["stratio"].mean())
    sd_stratio = float(df["stratio"].std())
    mean_testscr = float(df["testscr"].mean())
    sd_testscr = float(df["testscr"].std())

    # Correlation between ratio and test scores
    r, p_corr = stats.pearsonr(df["stratio"], df["testscr"])

    # Simple linear regression: testscr ~ stratio
    y = df["testscr"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    coef_stratio_simple = float(model_simple.params["stratio"])
    p_stratio_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key covariates
    covariates = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district income
        "feature13",  # % English learners
    ]
    available_covars = [c for c in covariates if c in df.columns]
    X_multi = sm.add_constant(df[["stratio"] + available_covars])
    model_multi = sm.OLS(y, X_multi).fit()
    coef_stratio_multi = float(model_multi.params["stratio"])
    p_stratio_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    # Determine overall answer and confidence
    negative_simple = coef_stratio_simple < 0
    negative_multi = coef_stratio_multi < 0
    consistent_negative = negative_simple and negative_multi and r < 0

    strong_significance = (
        p_corr < 0.01 and p_stratio_simple < 0.01 and p_stratio_multi < 0.01
    )
    moderate_significance = (
        p_corr < 0.05 and p_stratio_simple < 0.05 and p_stratio_multi < 0.05
    )

    if consistent_negative and strong_significance:
        response = "Yes"
        # Start from a high base confidence, adjust by effect size
        base_conf = 90
        effect_strength = min(10, int(abs(r) * 50))
        confidence = min(100, base_conf + effect_strength)
    elif consistent_negative and moderate_significance:
        response = "Yes"
        confidence = 80
    elif consistent_negative:
        response = "Yes"
        confidence = 70
    else:
        # Evidence is weak, inconsistent, or points in the opposite direction
        response = "No"
        confidence = 60

    # Build explanation text that matches the empirical results
    if abs(r) < 0.05 or p_corr >= 0.1:
        corr_phrase = (
            f"showed essentially no linear relationship (r = {r:.3f}, p {format_p(p_corr)})"
        )
    elif r < 0:
        corr_phrase = (
            f"was negative, meaning higher student–teacher ratios (more students per teacher) "
            f"were associated with lower scores (r = {r:.3f}, p {format_p(p_corr)})"
        )
    else:
        corr_phrase = (
            f"was positive, meaning higher student–teacher ratios were associated with higher "
            f"scores (r = {r:.3f}, p {format_p(p_corr)})"
        )

    def regression_phrase(coef: float, p_val: float, r2: float) -> str:
        if abs(coef) < 0.01 or p_val >= 0.1:
            return (
                f"the coefficient on the student–teacher ratio was very close to zero "
                f"({coef:.2f}) and not statistically significant (p {format_p(p_val)}, "
                f"R² = {r2:.3f}), indicating no meaningful association"
            )
        direction = "decrease" if coef < 0 else "increase"
        return (
            f"each additional student per teacher was associated with a "
            f"{abs(coef):.2f}-point {direction} in test scores "
            f"(p {format_p(p_val)}, R² = {r2:.3f})"
        )

    simple_phrase = regression_phrase(coef_stratio_simple, p_stratio_simple, r2_simple)
    multi_phrase = regression_phrase(coef_stratio_multi, p_stratio_multi, r2_multi)

    if response == "Yes":
        conclusion_phrase = (
            "the association between lower student–teacher ratios and higher academic "
            "performance is consistent in sign and statistically supported across models."
        )
    else:
        conclusion_phrase = (
            "the estimated effects of the student–teacher ratio are extremely small and not "
            "statistically distinguishable from zero in both simple and multiple regression "
            "models, so this dataset does not provide clear evidence that lower ratios are "
            "associated with higher academic performance."
        )

    explanation = (
        "I analyzed the 420 California K-6 and K-8 school districts using the caschools dataset. "
        f"I constructed a student–teacher ratio as total enrollment divided by the number of teachers "
        f"(mean {mean_stratio:.1f} students per teacher, SD {sd_stratio:.1f}) and an overall academic "
        f"performance score as the average of 5th-grade reading and math scores (mean {mean_testscr:.1f}, "
        f"SD {sd_testscr:.1f}). The Pearson correlation between the student–teacher ratio and test scores "
        f"{corr_phrase}. In a simple linear regression of scores on the ratio, {simple_phrase}. "
        f"In a multiple regression controlling for poverty (CalWorks and reduced-price lunch), computer "
        f"availability, per-pupil expenditure, district income, and the share of English learners, "
        f"{multi_phrase}. Based on these results, {conclusion_phrase}"
    )

    conclusion = {
        "response": response,
        "confidence": float(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
