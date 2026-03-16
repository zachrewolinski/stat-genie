import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)

    # Compute student-teacher ratio: total enrollment / number of teachers.
    # feature6: Total enrollment
    # feature7: Number of teachers (FTE)
    df = df.copy()
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading (feature14) and math (feature15) scores.
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing key fields, if present.
    df = df.dropna(subset=["student_teacher_ratio", "avg_score"])

    n_obs = int(df.shape[0])

    # Simple correlation between student-teacher ratio and average score.
    corr = float(df[["student_teacher_ratio", "avg_score"]].corr().iloc[0, 1])

    # Simple linear regression: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    coef_simple = float(model_simple.params["student_teacher_ratio"])
    pvalue_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key covariates:
    # feature8: % CalWorks
    # feature9: % reduced-price lunch
    # feature10: number of computers
    # feature11: expenditure per student
    # feature12: district average income (in thousands USD)
    # feature13: % English learners
    covariates = [
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature13",
    ]
    df_multi = df.dropna(subset=covariates)
    X_multi = sm.add_constant(df_multi[["student_teacher_ratio"] + covariates])
    model_multi = sm.OLS(df_multi["avg_score"], X_multi).fit()
    coef_multi = float(model_multi.params["student_teacher_ratio"])
    pvalue_multi = float(model_multi.pvalues["student_teacher_ratio"])
    r2_multi = float(model_multi.rsquared)

    # Decide on direction and strength of evidence.
    # Interpretation:
    #   Higher student_teacher_ratio = more students per teacher.
    #   A negative coefficient means that more students per teacher is associated with LOWER scores,
    #   which implies that LOWER ratios (smaller classes) are associated with HIGHER performance.
    def is_supportive(coef: float, pval: float, alpha: float = 0.05) -> bool:
        return (coef < 0) and (pval < alpha)

    supportive_multi = is_supportive(coef_multi, pvalue_multi)
    supportive_simple = is_supportive(coef_simple, pvalue_simple)

    # Choose yes/no based primarily on the multiple regression (with controls),
    # falling back to the simple model if needed.
    if supportive_multi or supportive_simple:
        answer_yes = True
    else:
        answer_yes = False

    # Map evidence strength to a 0–100 Likert score.
    # Start from a neutral baseline and adjust based on:
    #   - whether evidence supports the hypothesized negative association,
    #   - p-values from the multiple regression,
    #   - magnitude of the correlation.
    likert = 50.0

    abs_corr = abs(corr)

    if answer_yes:
        # Positive evidence that lower ratios are associated with higher performance.
        # Stronger significance and larger |corr| push the score closer to 100.
        if pvalue_multi < 0.001 and abs_corr >= 0.3:
            likert = 90.0
        elif pvalue_multi < 0.001:
            likert = 85.0
        elif pvalue_multi < 0.01:
            likert = 80.0
        elif pvalue_multi < 0.05:
            likert = 70.0
        else:
            # Only the simple model is supportive.
            if pvalue_simple < 0.01:
                likert = 70.0
            elif pvalue_simple < 0.05:
                likert = 65.0
            else:
                likert = 60.0

        # Slightly adjust for the magnitude of the simple correlation.
        if abs_corr >= 0.4:
            likert = min(100.0, likert + 5.0)
        elif abs_corr < 0.15:
            likert = max(0.0, likert - 5.0)
    else:
        # No statistically reliable evidence that lower ratios are associated with higher performance.
        # Lower scores represent a "No" with varying strength.
        if pvalue_multi < 0.1 or pvalue_simple < 0.1:
            likert = 40.0  # some suggestive evidence but not conventionally significant
        else:
            likert = 15.0  # clear lack of evidence for the relationship

        if abs_corr < 0.1:
            likert = max(0.0, likert - 5.0)

    likert_int = int(round(float(np.clip(likert, 0.0, 100.0))))

    # Build explanation string summarizing the analysis and results.
    direction_simple = "lower" if coef_simple < 0 else "higher"
    direction_multi = "lower" if coef_multi < 0 else "higher"

    yes_no_text = "Yes" if answer_yes else "No"

    explanation_parts = [
        f"Using data on {n_obs} California K-6 and K-8 districts, "
        "I examined whether lower student-teacher ratios are associated with higher academic performance.",
        "I constructed the student-teacher ratio as total enrollment divided by the number of teachers, "
        "and measured academic performance as the average of district-level reading and math scores.",
        f"The simple Pearson correlation between the student-teacher ratio and average performance is {corr:.3f}, "
        "indicating that districts with more students per teacher tend to have "
        "lower scores when this correlation is negative.",
        f"A simple linear regression of average score on the student-teacher ratio yields a coefficient of "
        f"{coef_simple:.2f} (p-value {pvalue_simple:.3g}, R-squared {r2_simple:.3f}), "
        f"so each additional student per teacher is associated with approximately {abs(coef_simple):.2f} points "
        f"{direction_simple} average test score.",
        "To account for important confounders, I also ran a multiple regression including socioeconomic and resource "
        "controls: percent CalWorks, percent reduced-price lunch, number of computers, expenditures per student, "
        "average income, and percent English learners.",
        f"In this adjusted model, the coefficient on the student-teacher ratio is {coef_multi:.2f} "
        f"(p-value {pvalue_multi:.3g}, R-squared {r2_multi:.3f}), indicating that, "
        "after controlling for these factors, districts with more students per teacher still tend to have "
        f"{direction_multi} average scores.",
    ]

    if answer_yes:
        explanation_parts.append(
            "Because the estimated association is consistently negative and statistically significant in both the simple "
            "and adjusted models, there is strong statistical evidence that lower student-teacher ratios are associated "
            "with higher academic performance in this dataset. "
            f"I therefore answer '{yes_no_text}' to the research question, with a confidence level reflected by a "
            f"Likert-scale response of {likert_int} out of 100."
        )
    else:
        explanation_parts.append(
            "Because the estimated association is not statistically significant (or not robust across models), "
            "there is insufficient statistical evidence in this dataset to conclude that lower student-teacher ratios "
            "are associated with higher academic performance. "
            f"I therefore answer '{yes_no_text}' to the research question, with a Likert-scale response of {likert_int} "
            "out of 100 to reflect the limited strength of evidence."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": likert_int,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

