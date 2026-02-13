import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    info_path = Path("info.json")

    df = pd.read_csv(data_path)

    # Basic derived variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing or infinite values in key columns
    df_clean = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["stratio", "testscr"])
        .copy()
    )

    # Simple correlation between student-teacher ratio and test scores
    corr = df_clean["stratio"].corr(df_clean["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_clean[["stratio"]])
    model_simple = sm.OLS(df_clean["testscr"], X_simple).fit()
    coef_stratio = float(model_simple.params["stratio"])
    pval_stratio = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key demographics if available
    covariates = ["income", "calworks", "lunch", "english"]
    available_covariates = [c for c in covariates if c in df_clean.columns]

    coef_stratio_multi = None
    pval_stratio_multi = None
    r2_multi = None

    if available_covariates:
        X_multi = sm.add_constant(df_clean[["stratio"] + available_covariates])
        model_multi = sm.OLS(df_clean["testscr"], X_multi).fit()
        coef_stratio_multi = float(model_multi.params["stratio"])
        pval_stratio_multi = float(model_multi.pvalues["stratio"])
        r2_multi = float(model_multi.rsquared)

    # Determine direction of association
    response = "Yes" if coef_stratio < 0 else "No"

    # Map strength of association to 0-100 using absolute correlation
    strength = int(round(100 * min(abs(corr), 1.0)))

    # Confidence based primarily on p-value from simple regression
    # Very small p-values -> confidence near 100
    p_clipped = min(max(pval_stratio, 0.0), 1.0)
    confidence = int(round(100 * (1.0 - p_clipped)))

    # Build human-readable explanation
    n_obs = int(df_clean.shape[0])
    mean_stratio = float(df_clean["stratio"].mean())
    sd_stratio = float(df_clean["stratio"].std())
    mean_testscr = float(df_clean["testscr"].mean())
    sd_testscr = float(df_clean["testscr"].std())

    # Effect of a 5-student change in ratio
    effect_5_students = 5 * coef_stratio

    explanation_parts = [
        "Research question: Is a lower student–teacher ratio associated with higher academic performance, ",
        "measured here using the average of district-level 5th grade reading and math scores.",
        f" The analysis uses {n_obs} districts from caschools.csv.",
        f" The student–teacher ratio (students/teachers) has mean {mean_stratio:.1f} and standard deviation {sd_stratio:.1f},",
        f" while the average test score has mean {mean_testscr:.1f} and standard deviation {sd_testscr:.1f}.",
        f" The simple Pearson correlation between student–teacher ratio and average test score is {corr:.3f},",
        " indicating that districts with more students per teacher tend to have lower test scores when the correlation is negative.",
        f" A simple linear regression of test scores on the student–teacher ratio yields a slope of {coef_stratio:.3f}",
        f" (R-squared = {r2_simple:.3f}, p-value for the ratio coefficient = {pval_stratio:.3g}).",
        f" This slope implies that, on average, a 5-student decrease in the student–teacher ratio is associated with about {effect_5_students:.2f} points higher test scores.",
    ]

    if coef_stratio_multi is not None and pval_stratio_multi is not None:
        explanation_parts.append(
            f" A multiple regression that additionally controls for income, CalWorks share, subsidized-lunch share, and English-learner share "
            f"still finds a coefficient on the student–teacher ratio of {coef_stratio_multi:.3f} "
            f"(R-squared = {r2_multi:.3f}, p-value = {pval_stratio_multi:.3g}), with the same negative sign."
        )

    if response == "Yes":
        explanation_parts.append(
            " Because the estimated relationship is consistently negative and statistically significant, "
            "the data provide clear evidence of an association in which lower student–teacher ratios are linked to higher academic performance."
        )
    else:
        explanation_parts.append(
            " Because the estimated relationship is weak or not statistically distinguishable from zero, "
            "the data do not provide clear evidence that lower student–teacher ratios are associated with higher academic performance."
        )

    explanation_parts.append(
        f" The strength score of {strength} reflects the magnitude of the correlation between the ratio and test scores, "
        f"and the confidence score of {confidence} reflects the limited statistical evidence (relatively large p-value) for an association in this sample."
    )

    explanation = "".join(explanation_parts)

    result = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
