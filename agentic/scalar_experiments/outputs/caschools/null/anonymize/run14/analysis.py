import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on the metadata in info.json:
    # - Student–teacher ratio: total enrollment / number of teachers
    # - Academic performance: average of reading and math scores
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop rows with any missing values in the variables used
    analysis_cols = [
        "student_teacher_ratio",
        "avg_score",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]
    data = df[analysis_cols].dropna().copy()

    # Simple Pearson correlation between student–teacher ratio and average score
    corr, corr_p = stats.pearsonr(
        data["student_teacher_ratio"], data["avg_score"]
    )

    # Multiple linear regression controlling for key covariates
    X = data[
        [
            "student_teacher_ratio",
            "feature8",
            "feature9",
            "feature11",
            "feature12",
            "feature13",
        ]
    ].copy()
    X = sm.add_constant(X)
    y = data["avg_score"]
    model = sm.OLS(y, X).fit()

    ratio_coef = float(model.params["student_teacher_ratio"])
    ratio_p = float(model.pvalues["student_teacher_ratio"])

    ratio_std = float(data["student_teacher_ratio"].std(ddof=1))
    score_std = float(data["avg_score"].std(ddof=1))

    # Standardized effect of student–teacher ratio on average score
    if ratio_std > 0 and score_std > 0:
        beta_std = ratio_coef * ratio_std / score_std
    else:
        beta_std = 0.0

    # Change in average score associated with a 1 SD decrease
    delta_score_per_sd_increase = ratio_coef * ratio_std
    delta_score_per_sd_decrease = -delta_score_per_sd_increase

    # Map evidence strength to a 0–100 Likert-scale "Yes" score.
    # We interpret "Yes" as lower student–teacher ratios being associated
    # with higher academic performance (i.e., negative coefficient on the ratio).
    if ratio_p >= 0.05 or ratio_coef >= 0:
        # No statistically significant evidence in the hypothesized direction
        response = 30
    else:
        # Significant association in the expected (negative) direction.
        # Use the standardized effect size to scale strength between 50 and 100.
        # Treat |beta_std| ~ 0.5 as a "large" effect.
        strength = min(1.0, max(0.0, abs(beta_std) / 0.5))
        response = int(round(50 + 50 * strength))

    # Build explanation text summarizing the analysis and evidence
    evidence_strength = (
        "strong"
        if response >= 80
        else "moderate"
        if response >= 60
        else "limited"
    )

    if ratio_p >= 0.05 or ratio_coef >= 0:
        association_summary = (
            f"The Pearson correlation between student–teacher ratio and average test score was "
            f"{corr:.3f} (p = {corr_p:.3g}), which is very close to zero and not statistically "
            "significant. In the multiple linear regression controlling for student demographics "
            "and resources, the coefficient on the student–teacher ratio was "
            f"{ratio_coef:.3f} (p = {ratio_p:.3g}), again indicating no statistically reliable "
            "association in the hypothesized (negative) direction. The implied change in average "
            "test scores for a one–standard-deviation decrease in the ratio is only "
            f"{delta_score_per_sd_decrease:.2f} points, which is substantively negligible."
        )
        conclusion_sentence = (
            "Taken together, these results do not provide evidence that lower student–teacher "
            "ratios are associated with higher academic performance in this dataset."
        )
    else:
        association_summary = (
            f"The Pearson correlation between student–teacher ratio and average test score was "
            f"{corr:.3f} (p = {corr_p:.3g}). In the multiple linear regression controlling for "
            "student demographics and resources, the coefficient on the student–teacher ratio was "
            f"{ratio_coef:.3f} (p = {ratio_p:.3g}), and a one–standard-deviation decrease in the "
            "ratio was associated with an estimated change of "
            f"{delta_score_per_sd_decrease:.2f} points in average test scores."
        )
        conclusion_sentence = (
            "These results indicate that districts with lower student–teacher ratios tend to have "
            "higher academic performance, although the analysis is observational and does not "
            "establish causality."
        )

    if response <= 50:
        final_sentence = (
            "Overall, the evidence supports a 'No' answer to the research question. "
            "On a 0–100 scale where 0 represents a strong 'No' and 100 represents a strong 'Yes', "
            f"this analysis corresponds to a response value of {response}."
        )
    else:
        final_sentence = (
            f"Overall, the evidence provides {evidence_strength} support for a 'Yes' answer to the "
            "research question. On a 0–100 scale where 0 represents a strong 'No' and 100 represents "
            f"a strong 'Yes', this analysis corresponds to a response value of {response}."
        )

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?\n\n"
        "Using the provided California school district data (420 K-6 and K-8 districts), "
        "I constructed a student–teacher ratio variable as total enrollment divided by the "
        "number of teachers (features 6 and 7). Academic performance was measured as the "
        "average of the district reading and math scores (features 14 and 15).\n\n"
        + association_summary
        + "\n\n"
        + conclusion_sentence
        + "\n\n"
        + final_sentence
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON object to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
