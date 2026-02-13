import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # Load research question metadata
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_questions = info.get("research_questions") or []
    research_question = research_questions[0] if research_questions else ""

    # Load dataset
    data_path = base_dir / "caschools.csv"
    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["reading_score"] = df["feature14"]
    df["math_score"] = df["feature15"]
    df["test_score"] = df[["reading_score", "math_score"]].mean(axis=1)

    df = df.dropna(
        subset=["student_teacher_ratio", "reading_score", "math_score", "test_score"]
    )

    # Descriptive statistics
    ratio = df["student_teacher_ratio"]
    test_score = df["test_score"]
    reading_score = df["reading_score"]
    math_score = df["math_score"]

    ratio_mean = float(ratio.mean())
    ratio_std = float(ratio.std())
    score_mean = float(test_score.mean())
    score_std = float(test_score.std())

    # Correlations
    corr_total = float(ratio.corr(test_score))
    corr_reading = float(ratio.corr(reading_score))
    corr_math = float(ratio.corr(math_score))

    # Simple linear regression: test score ~ student-teacher ratio
    X_simple = sm.add_constant(ratio)
    model_simple = sm.OLS(test_score, X_simple).fit()
    slope_simple = float(model_simple.params["student_teacher_ratio"])
    p_simple = float(model_simple.pvalues["student_teacher_ratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key controls:
    # expenditure per student (feature11),
    # district average income (feature12),
    # % on CalWorks (feature8),
    # % qualifying for reduced-price lunch (feature9),
    # % English learners (feature13)
    controls = ["feature11", "feature12", "feature8", "feature9", "feature13"]
    X_multi = df[["student_teacher_ratio"] + controls].copy()
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(test_score, X_multi).fit()

    slope_multi = float(model_multi.params["student_teacher_ratio"])
    p_multi = float(model_multi.pvalues["student_teacher_ratio"])
    r2_multi = float(model_multi.rsquared)

    # Effect of reducing the ratio by 5 students per teacher
    effect_per_5 = -5.0 * slope_multi

    # Map statistical evidence to a 0-100 Likert response
    response_score: float
    if slope_multi < 0:
        if p_multi < 0.001 and abs(corr_total) > 0.3:
            response_score = 90.0
        elif p_multi < 0.01 and abs(corr_total) > 0.25:
            response_score = 80.0
        elif p_multi < 0.05:
            response_score = 70.0
        elif p_multi < 0.1:
            response_score = 60.0
        else:
            response_score = 55.0
    elif slope_multi > 0:
        if p_multi < 0.05:
            response_score = 10.0
        else:
            response_score = 45.0
    else:
        response_score = 50.0

    response = int(max(0, min(100, round(response_score))))

    # Build explanation text
    explanation_parts = []
    if research_question:
        explanation_parts.append(f"Research question: {research_question}")
    else:
        explanation_parts.append(
            "Research question: Is a lower student-teacher ratio associated with higher academic performance?"
        )

    explanation_parts.append(
        "I analyzed data from 420 California K-6 and K-8 school districts that include enrollment, "
        "number of teachers, student demographics, and average 5th-grade reading and math scores."
    )
    explanation_parts.append(
        "I defined the student–teacher ratio as total enrollment (feature6) divided by number of teachers "
        "(feature7), and academic performance as the average of the reading (feature14) and math (feature15) "
        "scores for each district."
    )
    explanation_parts.append(
        f"The average student–teacher ratio is {ratio_mean:.1f} students per teacher (SD {ratio_std:.1f}), "
        f"and the average combined test score is {score_mean:.1f} (SD {score_std:.1f})."
    )
    corr_sentence = (
        "I first examined simple correlations between the student–teacher ratio and achievement measures. "
        f"The Pearson correlation between the ratio and the combined test score is {corr_total:.3f}, with "
        f"corresponding correlations of {corr_reading:.3f} for reading and {corr_math:.3f} for math."
    )
    explanation_parts.append(corr_sentence)
    if abs(corr_total) < 0.05:
        explanation_parts.append(
            "This correlation is very close to zero, indicating little to no linear relationship between "
            "class size and average test scores in this sample."
        )
    elif corr_total < 0:
        explanation_parts.append(
            "The negative correlation suggests that districts with smaller student–teacher ratios tend to have "
            "higher scores, although the magnitude of the association should be interpreted in light of the "
            "subsequent regression results."
        )
    else:
        explanation_parts.append(
            "The positive correlation suggests that districts with larger student–teacher ratios tend to have "
            "slightly higher scores, although the association is weak and must be interpreted cautiously."
        )

    explanation_parts.append(
        "Next, I fit a simple linear regression of the combined test score on the student–teacher ratio. "
        f"The estimated slope is {slope_simple:.2f} points per additional student per teacher "
        f"(R-squared = {r2_simple:.3f}, p-value = {p_simple:.3g})."
    )
    if p_simple < 0.05:
        if slope_simple < 0:
            explanation_parts.append(
                "In this simple model, the negative and statistically significant slope indicates that larger "
                "classes are associated with lower average test scores."
            )
        else:
            explanation_parts.append(
                "In this simple model, the positive and statistically significant slope indicates that larger "
                "classes are associated with slightly higher average test scores, contrary to the original "
                "hypothesis."
            )
    else:
        explanation_parts.append(
            "However, this slope is not statistically distinguishable from zero at conventional levels, so the "
            "simple regression does not provide strong evidence of a linear association between class size and "
            "test scores."
        )

    explanation_parts.append(
        "To account for potential confounding, I then estimated a multiple regression including expenditure per "
        "student (feature11), district average income (feature12), and the percentages of students on CalWorks "
        "(feature8), qualifying for reduced-price lunch (feature9), and English learners (feature13)."
    )
    explanation_parts.append(
        f"In this adjusted model, the slope on the student–teacher ratio is {slope_multi:.2f} points per "
        f"additional student per teacher (R-squared = {r2_multi:.3f}, p-value = {p_multi:.3g}). "
        f"Reducing the ratio by 5 students per teacher is associated with an estimated change of about "
        f"{effect_per_5:.1f} points in the average test score, holding these demographic and resource variables "
        "constant (positive values correspond to higher scores when classes are smaller)."
    )
    if p_multi < 0.05:
        if slope_multi < 0:
            explanation_parts.append(
                "The adjusted results therefore support the hypothesis that, after accounting for observed "
                "socioeconomic and resource differences, districts with smaller student–teacher ratios tend to "
                "have higher academic performance."
            )
        else:
            explanation_parts.append(
                "The adjusted results suggest that, after accounting for observed socioeconomic and resource "
                "differences, districts with larger student–teacher ratios tend to have slightly higher scores; "
                "however, this pattern must be interpreted with caution."
            )
    else:
        explanation_parts.append(
            "Because the adjusted slope is very close to zero and not statistically significant, the multiple "
            "regression does not provide strong evidence that class size is systematically related to average "
            "test scores once these covariates are taken into account."
        )

    if response >= 60:
        conclusion_sentence = (
            "Overall, the balance of evidence from the correlations and regression models supports a 'Yes' "
            "answer to the research question: lower student–teacher ratios are associated with higher academic "
            "performance in this dataset. "
            f"I therefore place my answer at {response} on a 0–100 scale, where higher values represent stronger "
            "support for a 'Yes' answer."
        )
    elif response <= 40:
        conclusion_sentence = (
            "Overall, the analyses do not provide convincing evidence that lower student–teacher ratios are "
            "associated with higher academic performance in this dataset; if anything, the estimated "
            "associations are extremely small and imprecise. "
            f"I therefore place my answer at {response} on a 0–100 scale, below the neutral midpoint, to reflect "
            "a lean toward a 'No' answer."
        )
    else:
        conclusion_sentence = (
            "Overall, the analyses suggest little to no systematic relationship between student–teacher ratios "
            "and average test scores in this dataset. "
            f"I therefore place my answer at {response} on a 0–100 scale, near the middle of the range, to "
            "reflect the limited evidence for either a clear 'Yes' or 'No' answer."
        )
    explanation_parts.append(conclusion_sentence)

    explanation = "\n\n".join(explanation_parts)

    result = {"response": response, "explanation": explanation}

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
