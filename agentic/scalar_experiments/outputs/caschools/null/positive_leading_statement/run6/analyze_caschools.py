import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "caschools.csv"

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(
        subset=["stratio", "avg_score", "calworks", "lunch", "income", "english"]
    ).copy()

    # Simple correlation between student-teacher ratio and average test score
    corr, corr_p = pearsonr(df["stratio"], df["avg_score"])

    # Bivariate regression: avg_score ~ stratio
    model_simple = smf.ols("avg_score ~ stratio", data=df).fit()

    # Multivariate regression controlling for key demographics
    model_multi = smf.ols(
        "avg_score ~ stratio + calworks + lunch + income + english", data=df
    ).fit()

    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    coef_multi = float(model_multi.params["stratio"])
    p_multi = float(model_multi.pvalues["stratio"])

    r2_simple = float(model_simple.rsquared)
    r2_multi = float(model_multi.rsquared)

    # Effect size: change in average test score associated with a 1-SD change
    sd_stratio = float(df["stratio"].std())
    sd_effect_multi = coef_multi * sd_stratio

    # Determine whether the evidence supports a negative relationship:
    # lower student-teacher ratios (fewer students per teacher) associated with higher scores.
    support_direction = (coef_simple < 0) and (coef_multi < 0) and (corr < 0)

    # Significance levels
    sig_strict = (p_simple < 0.01) and (p_multi < 0.01)
    sig_moderate = (p_simple < 0.05) and (p_multi < 0.05)
    sig_weak = (p_simple < 0.1) or (p_multi < 0.1)

    # Map statistical evidence to a 0–100 Likert scale.
    if support_direction:
        if sig_strict and abs(sd_effect_multi) >= 10:
            response = 90
        elif sig_strict and abs(sd_effect_multi) >= 5:
            response = 85
        elif sig_moderate and abs(sd_effect_multi) >= 5:
            response = 80
        elif sig_moderate:
            response = 70
        elif sig_weak:
            response = 60
        else:
            response = 55
    else:
        if sig_strict:
            response = 10
        elif sig_moderate:
            response = 20
        elif sig_weak:
            response = 30
        else:
            response = 40

    if support_direction and (p_simple < 0.05) and (p_multi < 0.05):
        yes_no_text = "Yes"
        qualitative = (
            "There is consistent evidence that districts with lower student-teacher "
            "ratios tend to have higher average test scores."
        )
    elif support_direction:
        yes_no_text = "Yes (but with modest statistical strength)"
        qualitative = (
            "The direction of the association is consistent with lower student-teacher "
            "ratios being linked to higher scores, but the statistical evidence is only "
            "modest."
        )
    else:
        yes_no_text = "No"
        qualitative = (
            "The data do not provide strong, consistent evidence that lower student-teacher "
            "ratios are associated with higher average test scores once we examine the "
            "relationship statistically."
        )

    explanation = (
        f"Research question: Is a lower student-teacher ratio associated with higher academic "
        f"performance?\n\n"
        f"Using data from 420 California K-6 and K-8 districts, I constructed the student-teacher "
        f"ratio as students divided by teachers and an academic performance measure as the average "
        f"of district-level reading and math scores.\n\n"
        f"Bivariate evidence: The Pearson correlation between the student-teacher ratio and the "
        f"average test score is {corr:.2f} (p-value = {corr_p:.3g}), indicating that districts "
        f"with higher ratios (more students per teacher) tend to have "
        f"{'lower' if corr < 0 else 'higher'} scores.\n\n"
        f"Regression evidence: In a simple linear regression of average score on the student-teacher "
        f"ratio, the coefficient on the ratio is {coef_simple:.2f} (p-value = {p_simple:.3g}, "
        f"R-squared = {r2_simple:.2f}). In a multiple regression that additionally controls for "
        f"CalWorks participation, reduced-price lunch eligibility, district income, and the share "
        f"of English learners, the coefficient on the student-teacher ratio is {coef_multi:.2f} "
        f"(p-value = {p_multi:.3g}, R-squared = {r2_multi:.2f}). A one–standard-deviation increase "
        f"in the student-teacher ratio is associated with an estimated change of "
        f"{sd_effect_multi:.2f} points in the average test score in the multivariate model.\n\n"
        f"Overall assessment: {qualitative} Based on these results, my overall answer to the "
        f"research question is: {yes_no_text}. On a 0–100 scale where 0 represents a strong 'No' "
        f"and 100 represents a strong 'Yes', I assign a value of {int(response)} to the strength "
        f"of this conclusion."
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

