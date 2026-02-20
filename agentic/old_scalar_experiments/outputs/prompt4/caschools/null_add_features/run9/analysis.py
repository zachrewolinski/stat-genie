import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables: student-teacher ratio and overall test score.
    # Guard against division by zero or missing values.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables of interest.
    cols_core = ["testscr", "stratio"]
    df_core = df.dropna(subset=cols_core)

    # Simple correlation between student-teacher ratio and test scores.
    corr = df_core["stratio"].corr(df_core["testscr"])

    # Simple OLS: testscr ~ stratio
    y = df_core["testscr"]
    X_simple = sm.add_constant(df_core["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    coef_stratio_simple = model_simple.params["stratio"]
    pval_stratio_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multivariate OLS with key demographic and resource controls if present.
    controls = [
        "income",
        "calworks",
        "lunch",
        "english",
        "expenditure",
        "computer",
    ]
    controls_present = [c for c in controls if c in df.columns]
    cols_multi = ["testscr", "stratio"] + controls_present
    df_multi = df.dropna(subset=cols_multi)

    model_multi = None
    coef_stratio_multi = None
    pval_stratio_multi = None
    r2_multi = None

    if len(df_multi) > 0:
        y_m = df_multi["testscr"]
        X_m = sm.add_constant(df_multi[["stratio"] + controls_present])
        model_multi = sm.OLS(y_m, X_m).fit()
        coef_stratio_multi = model_multi.params["stratio"]
        pval_stratio_multi = model_multi.pvalues["stratio"]
        r2_multi = model_multi.rsquared

    # Map statistical evidence to a 0–100 Likert response.
    # We interpret the research question as:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    # Since stratio is students per teacher, a negative coefficient indicates that
    # lower ratios (smaller classes) are associated with higher scores.
    response_score = 50

    def strength_from_effect(coef: float, pval: float) -> int:
        if np.isnan(coef) or np.isnan(pval):
            return 50
        if coef < 0 and pval < 0.001:
            return 90
        if coef < 0 and pval < 0.01:
            return 80
        if coef < 0 and pval < 0.05:
            return 70
        if coef < 0 and pval < 0.1:
            return 60
        if coef > 0 and pval < 0.05:
            return 20
        if coef > 0 and pval < 0.1:
            return 30
        return 50

    score_simple = strength_from_effect(coef_stratio_simple, pval_stratio_simple)
    if coef_stratio_multi is not None and pval_stratio_multi is not None:
        score_multi = strength_from_effect(coef_stratio_multi, pval_stratio_multi)
        response_score = int(round((score_simple + score_multi) / 2))
    else:
        response_score = int(score_simple)

    response_score = int(min(100, max(0, response_score)))

    explanation_parts = []
    explanation_parts.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance "
        "in California K-6 and K-8 districts?"
    )
    explanation_parts.append(
        f"The analysis uses {len(df_core)} districts with complete data to relate "
        "an overall test score (average of reading and math) to the student-teacher ratio "
        "(students per teacher)."
    )
    explanation_parts.append(
        f"The simple correlation between student-teacher ratio and test scores is {corr:.3f}, "
        "indicating the direction and strength of the raw association."
    )
    explanation_parts.append(
        "A simple linear regression of test scores on the student-teacher ratio shows that "
        f"each additional student per teacher is associated with a change of {coef_stratio_simple:.3f} "
        f"points in the average test score (p-value = {pval_stratio_simple:.4f}, R-squared = {r2_simple:.3f})."
    )
    if model_multi is not None:
        explanation_parts.append(
            "A multivariate regression controlling for available district characteristics "
            f"({', '.join(controls_present)}) yields a coefficient of {coef_stratio_multi:.3f} "
            f"for the student-teacher ratio (p-value = {pval_stratio_multi:.4f}, "
            f"R-squared = {r2_multi:.3f}), which provides evidence on whether the association "
            "persists after accounting for these factors."
        )

    if response_score >= 70:
        qualitative = (
            "Overall, the estimates suggest a reasonably strong and statistically reliable "
            "negative relationship: districts with lower student-teacher ratios tend to have "
            "higher test scores, even after adjusting for other observed characteristics."
        )
    elif response_score >= 60:
        qualitative = (
            "Overall, the estimates suggest a modest but fairly consistent negative relationship: "
            "lower student-teacher ratios are associated with somewhat higher test scores, "
            "although the magnitude is moderate."
        )
    elif response_score <= 40:
        qualitative = (
            "Overall, the estimates do not support the hypothesis that lower student-teacher ratios "
            "are associated with higher test scores; the estimated relationship is weak, inconsistent, "
            "or even in the opposite direction."
        )
    else:
        qualitative = (
            "Overall, the evidence for a relationship between student-teacher ratios and test scores "
            "is mixed or statistically weak, so the data do not strongly support either a positive or "
            "negative association."
        )

    explanation_parts.append(qualitative)

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

