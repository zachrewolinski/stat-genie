import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (for context only; we don't programmatically branch on it)
    info_path = base_path / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load data
    data_path = base_path / "caschools.csv"
    df = pd.read_csv(data_path)

    # Construct key variables
    # Student-teacher ratio: students per teacher (higher = larger classes)
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing key variables, if any
    df_model = df[["testscr", "stratio", "income", "english", "lunch"]].dropna()

    # Simple bivariate Pearson correlation between class size and performance
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Multivariable OLS regression controlling for key demographics
    X = df_model[["stratio", "income", "english", "lunch"]]
    X = sm.add_constant(X)
    y = df_model["testscr"]
    model = sm.OLS(y, X).fit()

    coef_str = model.params["stratio"]
    p_str = model.pvalues["stratio"]

    # Summary statistics for interpretation
    n_obs = int(df_model.shape[0])
    mean_str = float(df_model["stratio"].mean())
    std_str = float(df_model["stratio"].std())
    mean_testscr = float(df_model["testscr"].mean())
    std_testscr = float(df_model["testscr"].std())
    r_squared = float(model.rsquared)

    # Translate statistical evidence into a 0–100 Likert-style response
    # We consider: sign and magnitude of coef_str, its p-value, and correlation strength.
    # Negative coef (higher ratios -> lower scores) aligns with "Yes" to the question.
    if p_str < 0.001 and coef_str < 0:
        base_score = 90
    elif p_str < 0.01 and coef_str < 0:
        base_score = 80
    elif p_str < 0.05 and coef_str < 0:
        base_score = 70
    elif p_str < 0.05 and coef_str > 0:
        # Significant but in the opposite direction of the hypothesised relationship.
        base_score = 20
    else:
        # No strong evidence for a relationship in either direction.
        base_score = 40

    # Adjust modestly based on correlation magnitude
    corr_strength = abs(r)
    if corr_strength > 0.4:
        base_score += 5
    elif corr_strength < 0.1:
        base_score -= 5

    # Clip to [0, 100] and round to int
    response_score = int(min(100, max(0, round(base_score))))

    # Build explanation text
    direction_text = (
        "a negative association where districts with larger student–teacher ratios "
        "tend to have lower test scores"
        if coef_str < 0
        else "a positive association where districts with larger student–teacher ratios "
        "tend to have higher test scores"
    )

    significance_text = (
        f"The coefficient on the student–teacher ratio is {coef_str:.3f} "
        f"with a p-value of {p_str:.3g}, and the model R-squared is {r_squared:.3f}."
    )

    corr_text = (
        f"The Pearson correlation between the student–teacher ratio and the composite "
        f"test score is r = {r:.3f} (p = {p_corr:.3g})."
    )

    context_text = (
        f"The analysis uses data from {n_obs} California K–6/K–8 districts. "
        f"The mean student–teacher ratio is {mean_str:.1f} (SD = {std_str:.1f}), "
        f"and the mean composite test score is {mean_testscr:.1f} (SD = {std_testscr:.1f})."
    )

    if coef_str < 0 and p_str < 0.05:
        answer_summary = (
            "Overall, these results provide statistically significant evidence that districts "
            "with lower student–teacher ratios tend to have higher academic performance, "
            "even after controlling for income, the share of English learners, and the "
            "percentage of students on subsidized lunch."
        )
    elif coef_str > 0 and p_str < 0.05:
        answer_summary = (
            "Overall, the evidence points to a statistically significant association in the "
            "opposite direction of the original hypothesis: districts with lower student–teacher "
            "ratios tend to have slightly lower test scores after controlling for covariates."
        )
    else:
        answer_summary = (
            "Overall, the evidence for a systematic relationship between the student–teacher ratio "
            "and academic performance is weak once key demographic controls are included."
        )

    explanation = (
        f"Research question: {research_question}\n\n"
        f"{context_text}\n\n"
        f"{corr_text} This indicates {direction_text} in the raw data.\n\n"
        f"In a multivariable OLS regression of the composite test score on the student–teacher "
        f"ratio, district income, the percentage of English learners, and the percentage of "
        f"students eligible for reduced-price lunch, {significance_text} "
        f"{answer_summary}\n\n"
        f"The 0–100 response score of {response_score} reflects the strength and direction of "
        f"this evidence, where higher values correspond to stronger support for the statement "
        f"that lower student–teacher ratios are associated with higher academic performance."
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

