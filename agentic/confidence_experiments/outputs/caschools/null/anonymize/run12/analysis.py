import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def map_to_likert(
    corr: float,
    slope_simple: float,
    pval_simple: float,
    r2_simple: float,
    slope_adjusted: float,
    pval_adjusted: float,
    r2_adjusted: float,
) -> int:
    """
    Map strength and robustness of the association to a 0–100 Likert score.

    0 = strong "No", 100 = strong "Yes".
    """
    # Check that the relationship is in the expected direction:
    # more students per teacher (higher ratio) -> lower performance (negative slope and correlation).
    expected_direction = corr < 0 and slope_simple < 0 and slope_adjusted < 0

    # Require reasonably strong statistical evidence in both simple and adjusted models.
    strong_significance = pval_simple < 0.01 and pval_adjusted < 0.01

    if not expected_direction or not strong_significance:
        # Weak or inconsistent evidence for the hypothesized relationship.
        return 20

    # Combine simple and adjusted model fit into a single strength metric.
    avg_r2 = (r2_simple + r2_adjusted) / 2.0
    # Use both correlation and model fit as indicators of strength.
    strength_raw = max(abs(corr), np.sqrt(max(avg_r2, 0.0)))

    # Cap strength at a value considered "strong" around 0.5.
    strength_capped = min(1.0, strength_raw / 0.5)

    # Map [0, 1] strength to a Likert band [60, 90] for "Yes" answers.
    base = 60
    max_extra = 30
    score = base + max_extra * strength_capped

    return int(round(score))


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("caschools.csv")

    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables:
    # - student-teacher ratio = total enrollment / number of teachers
    # - academic performance = average of reading and math scores
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["test_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Demographic and resource controls
    controls = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
        "feature6",   # total enrollment
    ]

    cols = ["student_teacher_ratio", "test_score"] + controls
    data = df[cols].dropna()

    # Simple correlation
    corr = data["student_teacher_ratio"].corr(data["test_score"])

    # Simple linear regression: test_score ~ student_teacher_ratio
    X_simple = sm.add_constant(data["student_teacher_ratio"])
    model_simple = sm.OLS(data["test_score"], X_simple).fit()
    slope_simple = float(model_simple.params["student_teacher_ratio"])
    pval_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with controls
    X_adjusted = sm.add_constant(data[["student_teacher_ratio"] + controls])
    model_adjusted = sm.OLS(data["test_score"], X_adjusted).fit()
    slope_adjusted = float(model_adjusted.params["student_teacher_ratio"])
    pval_adjusted = float(model_adjusted.pvalues["student_teacher_ratio"])
    r2_adjusted = float(model_adjusted.rsquared)

    response_score = map_to_likert(
        corr=corr,
        slope_simple=slope_simple,
        pval_simple=pval_simple,
        r2_simple=r2_simple,
        slope_adjusted=slope_adjusted,
        pval_adjusted=pval_adjusted,
        r2_adjusted=r2_adjusted,
    )

    # Determine verbal summary based on the numerical results.
    answer_word = "Yes" if response_score >= 50 else "No"

    if pval_simple < 0.05:
        simple_sig_sentence = (
            "The slope is statistically different from zero at conventional significance levels, "
            "indicating a measurable linear association in the bivariate model."
        )
    else:
        simple_sig_sentence = (
            "The slope is not statistically different from zero at conventional significance levels, "
            "so the bivariate model does not provide strong evidence of a linear association."
        )

    if pval_adjusted < 0.05:
        adjusted_sig_sentence = (
            "After controlling for these covariates, the coefficient on student–teacher ratio remains statistically "
            "significant, so the association appears robust to these adjustments."
        )
    else:
        adjusted_sig_sentence = (
            "After controlling for these covariates, the coefficient on student–teacher ratio is not statistically "
            "significant, so the adjusted model also does not show strong evidence of a linear association."
        )

    if answer_word == "Yes":
        conclusion_sentence = (
            "Taken together, these results provide evidence that lower student–teacher ratios are associated with higher "
            "academic performance at the district level in this dataset. The relationship is not deterministic—other "
            "factors also matter—but the direction and statistical significance are consistent. "
            f"Accordingly, I answer 'Yes' to the research question and place my confidence at {response_score} on a 0–100 "
            "Likert scale, where higher values represent stronger evidence for a positive association between lower "
            "student–teacher ratios and higher academic performance."
        )
    else:
        conclusion_sentence = (
            "Taken together, these results provide little evidence that lower student–teacher ratios are associated with "
            "higher academic performance at the district level in this dataset. The estimates are small in magnitude and "
            "statistically indistinguishable from zero, so any true association—if it exists—is likely to be weak in "
            "this sample. Accordingly, I answer 'No' to the research question and place my confidence at "
            f"{response_score} on a 0–100 Likert scale, where 0 corresponds to a strong 'No' answer and 100 corresponds "
            "to a strong 'Yes' answer."
        )

    explanation = (
        f"Research question: {research_question}\n\n"
        "I used district-level data on enrollment, number of teachers, and standardized test scores.\n"
        "Student–teacher ratio was defined as total enrollment divided by the number of full-time-equivalent teachers, "
        "so a lower ratio corresponds to fewer students per teacher. Academic performance was summarized as the "
        "average of the district reading and math scores.\n\n"
        f"The simple Pearson correlation between student–teacher ratio and average test score is {corr:.3f}. Values "
        "near 0 indicate little or no linear association, whereas values near -1 or 1 indicate a strong relationship.\n"
        f"In a simple linear regression of test scores on student–teacher ratio, the estimated slope is {slope_simple:.3f} "
        f"points per one-student increase in the ratio (R² = {r2_simple:.3f}, p-value = {pval_simple:.3e}). "
        f"{simple_sig_sentence}\n"
        f"When I control for demographic and resource variables (percent CalWorks, percent reduced-price lunch, "
        f"number of computers, expenditure per student, district average income, percent English learners, and total "
        f"enrollment), the coefficient on student–teacher ratio remains {slope_adjusted:.3f} with p-value "
        f"{pval_adjusted:.3e} and model R² = {r2_adjusted:.3f}. {adjusted_sig_sentence}\n\n"
        f"{conclusion_sentence}"
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
