import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # According to info.json metadata:
    # - "english" column: total enrollment
    # - "students" column: number of teachers
    # - "district" column: average reading score
    # - "expenditure" column: average math score

    # Construct student-teacher ratio (students per teacher)
    df = df.copy()
    df["student_teacher_ratio"] = df["english"] / df["students"]

    # Overall academic performance: average of reading and math scores
    df["avg_performance"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with any missing values in key variables
    subset = df[["student_teacher_ratio", "avg_performance"]].dropna()

    # Compute Pearson correlation and simple linear regression
    x = subset["student_teacher_ratio"].to_numpy()
    y = subset["avg_performance"].to_numpy()

    r, p_value = stats.pearsonr(x, y)

    # Simple linear regression: avg_performance = beta0 + beta1 * ratio
    slope, intercept, r_value, p_reg, stderr = stats.linregress(x, y)

    # Interpretation: negative slope / correlation means that lower ratio
    # (fewer students per teacher) is associated with higher performance.
    association_strength = abs(r)
    is_negative = slope < 0

    # Map evidence to a 0–100 Likert scale.
    # Heuristic:
    # - Start from base = 50 (neutral).
    # - Add up to 40 points for strength of correlation |r| (capped at 1).
    # - Add 10 points if the relationship is in the hypothesized (negative) direction
    #   and statistically significant at 1% level.
    base = 50
    strength_component = min(association_strength, 1.0) * 40
    significance_bonus = 0
    if is_negative and p_value < 0.01:
        significance_bonus = 10

    score_float = base + strength_component + significance_bonus
    score = int(round(max(0, min(100, score_float))))

    # Build human-readable explanation
    direction_text = (
        "lower student-teacher ratios are associated with higher average test scores"
        if is_negative
        else "higher student-teacher ratios are associated with higher average test scores"
    )

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?\n"
        "Using the provided California school district data, I constructed the student-teacher ratio as total "
        "enrollment divided by the number of teachers (columns 'english' and 'students'), and defined academic "
        "performance as the average of district-level reading and math scores (columns 'district' and 'expenditure').\n"
        f"The Pearson correlation between the student-teacher ratio and average performance is r = {r:.3f} "
        f"with p-value {p_value:.3g}. The linear regression slope is {slope:.3f}, meaning that, on average, a one-unit "
        "increase in the student-teacher ratio is associated with a change of that many points in the combined test score. "
        f"In this dataset, {direction_text}, and this relationship is "
        f"{'statistically significant' if p_value < 0.05 else 'not statistically strong'} "
        "at conventional levels. "
        f"Based on the magnitude and direction of the association (|r| = {association_strength:.3f}) "
        "and its statistical significance, I place my answer at "
        f"{score}/100 on a scale where 0 is a strong 'No' and 100 is a strong 'Yes'."
    )

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    # Write required output file with ONLY the JSON object
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

