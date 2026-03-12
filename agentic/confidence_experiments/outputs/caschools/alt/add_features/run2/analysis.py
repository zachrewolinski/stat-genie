import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    base_dir = Path(__file__).parent

    info = load_metadata(base_dir / "info.json")
    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(base_dir / "caschools.csv")

    # Construct key variables: student-teacher ratio and test score.
    df = df.copy()
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used.
    cols_basic = ["testscr", "str"]
    cols_controls = cols_basic + [
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_basic = df[cols_basic].dropna()
    df_full = df[cols_controls].dropna()

    # 1) Correlation between student-teacher ratio and test scores.
    r_pearson, p_pearson = stats.pearsonr(df_basic["str"], df_basic["testscr"])

    # 2) Simple OLS: testscr ~ str
    X_basic = sm.add_constant(df_basic["str"])
    y_basic = df_basic["testscr"]
    model_basic = sm.OLS(y_basic, X_basic).fit()
    coef_str_basic = model_basic.params["str"]
    se_str_basic = model_basic.bse["str"]
    p_str_basic = model_basic.pvalues["str"]
    ci_basic_low, ci_basic_high = model_basic.conf_int().loc["str"]
    r2_basic = float(model_basic.rsquared)

    # 3) Multiple OLS with demographic and resource controls.
    X_full = sm.add_constant(df_full[["str", "income", "english", "lunch", "calworks", "expenditure", "computer"]])
    y_full = df_full["testscr"]
    model_full = sm.OLS(y_full, X_full).fit()
    coef_str_full = model_full.params["str"]
    se_str_full = model_full.bse["str"]
    p_str_full = model_full.pvalues["str"]
    ci_full_low, ci_full_high = model_full.conf_int().loc["str"]
    r2_full = float(model_full.rsquared)

    # Synthesize evidence into a Likert-style strength score (0-100).
    # Start from 50 (uncertain) and adjust based on:
    # - Sign of association (negative r and coefficients support "lower STR -> higher performance").
    # - Magnitude of correlation.
    # - Statistical significance in both models.
    score = 50

    # Direction: strongly negative relationships increase the score.
    if coef_str_full < 0 and coef_str_basic < 0 and r_pearson < 0:
        score += 20
    elif (coef_str_full < 0 and coef_str_basic < 0) or r_pearson < 0:
        score += 10
    else:
        score -= 10

    # Strength of correlation.
    abs_r = abs(r_pearson)
    if abs_r >= 0.5:
        score += 15
    elif abs_r >= 0.3:
        score += 10
    elif abs_r >= 0.1:
        score += 5
    else:
        score -= 5

    # Statistical significance of STR coefficient in full model.
    if p_str_full < 0.001:
        score += 15
    elif p_str_full < 0.01:
        score += 10
    elif p_str_full < 0.05:
        score += 5
    else:
        score -= 10

    # Keep score in [0, 100] and cast to int.
    score = int(np.clip(score, 0, 100))

    # Build human-readable explanation with key statistics.
    direction_text = "negative" if coef_str_full < 0 else "positive"
    significance_text = "highly statistically significant" if p_str_full < 0.001 else (
        "statistically significant" if p_str_full < 0.05 else "not statistically significant"
    )

    explanation_lines = [
        f"Research question: {research_question}",
        "Using data on 420 California school districts, I examined whether a lower student–teacher ratio "
        "is associated with higher academic performance (average of reading and math test scores).",
        f"The simple correlation between student–teacher ratio and average test score is {r_pearson:.3f} "
        f"(p-value = {p_pearson:.3g}), indicating a modest {('negative' if r_pearson < 0 else 'positive')} association.",
        f"In a simple OLS regression of test scores on the student–teacher ratio, the coefficient on the ratio is "
        f"{coef_str_basic:.3f} (SE = {se_str_basic:.3f}, 95% CI [{ci_basic_low:.3f}, {ci_basic_high:.3f}], "
        f"p-value = {p_str_basic:.3g}), with R-squared = {r2_basic:.3f}.",
        "This coefficient measures the expected change in average test score (points) for a one-student increase "
        "in the number of students per teacher.",
        f"Adding controls for income, English-learner share, reduced-price-lunch share, CalWorks participation, "
        f"expenditures per student, and computers per classroom, the coefficient on student–teacher ratio remains "
        f"{direction_text}: {coef_str_full:.3f} (SE = {se_str_full:.3f}, 95% CI [{ci_full_low:.3f}, {ci_full_high:.3f}], "
        f"p-value = {p_str_full:.3g}), with R-squared = {r2_full:.3f}.",
        f"This controlled association is {significance_text}, providing evidence that districts with lower student–teacher "
        "ratios tend to have higher test scores, even after adjusting for these observable characteristics.",
        f"Based on the magnitude and statistical significance of the estimated relationships, I rate the evidence that "
        f"lower student–teacher ratios are associated with higher academic performance as {score}/100 on a Likert scale, "
        "where higher values indicate stronger support for a 'Yes' answer.",
    ]

    explanation = "\n".join(explanation_lines)

    conclusion = {"response": score, "explanation": explanation}

    # Write the required JSON object to conclusion.txt
    out_path = base_dir / "conclusion.txt"
    with out_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

