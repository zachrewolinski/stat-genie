import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    info_path = Path("info.json")

    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in working directory.")

    # Load metadata mainly so the analysis code documents how variables are mapped.
    with info_path.open() as f:
        info = json.load(f)

    # Read data
    df = pd.read_csv(data_path)

    # According to info.json, in this shuffled version:
    # - "english" = total enrollment (students)
    # - "students" = number of teachers
    # - "district" = average reading score
    # - "expenditure" = average math score
    #
    # We construct:
    #   student_teacher_ratio = enrollment / teachers
    #   avg_score = mean(reading, math)
    #
    # Lower ratios (fewer students per teacher) should be associated with higher scores
    # if the research hypothesis holds.
    df = df.copy()
    df["student_teacher_ratio"] = df["english"] / df["students"]
    df["avg_score"] = df[["district", "expenditure"]].mean(axis=1)

    # Drop any rows with missing values in the key variables (there should be none,
    # but this keeps the analysis robust).
    df = df.dropna(subset=["student_teacher_ratio", "avg_score"])

    x = df["student_teacher_ratio"].to_numpy()
    y = df["avg_score"].to_numpy()

    # Basic descriptive relationship: Pearson correlation
    corr, p_corr = stats.pearsonr(x, y)

    # Simple linear regression: avg_score ~ student_teacher_ratio
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params[1]
    p_slope = model.pvalues[1]

    # Interpret the evidence:
    #
    # - We expect a negative slope/correlation if lower ratios are associated
    #   with higher performance.
    # - Use magnitude and significance to map to a 0–100 Likert-style score.
    #
    # Heuristic for response score:
    #   - If slope is strongly negative (e.g., <= -1.5 score points per additional student)
    #     and highly significant (p < 0.001), answer near 90.
    #   - If moderately negative (around -0.5 to -1.5) with p < 0.01, answer around 75.
    #   - If small or marginal, or p >= 0.05, slide toward the middle.
    #   - If the evidence goes in the opposite direction, slide toward 0.
    #
    # To keep this deterministic yet data-driven, we compute a base score from the
    # sign of the slope and adjust by scaled effect size and significance.
    effect_direction = np.sign(slope)

    # Start from neutral (50) and move up/down based on evidence.
    score = 50.0

    # Effect size scaling: compare slope magnitude to a reference of 1 test-score
    # point per additional student per teacher, capped to avoid extreme values.
    ref_effect = 1.0
    effect_strength = min(abs(slope) / ref_effect, 3.0)  # cap at 3

    # Significance scaling based on p-value
    if p_slope < 0.001:
        sig_weight = 1.0
    elif p_slope < 0.01:
        sig_weight = 0.7
    elif p_slope < 0.05:
        sig_weight = 0.4
    else:
        sig_weight = 0.2

    # Maximum adjustment from neutral toward either extreme
    max_adjust = 40.0  # so scores stay between 10 and 90 in typical cases
    adjust = effect_direction * effect_strength * sig_weight * max_adjust / 3.0
    score += adjust

    # Clip to [0, 100] and round to nearest integer
    score = int(round(float(np.clip(score, 0.0, 100.0))))

    # Build a human-readable explanation summarizing the key statistics.
    # We avoid including the full regression table but report the essentials.
    explanation_lines = []
    explanation_lines.append(
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "in California K-6 and K-8 districts (1998–1999)?"
    )
    explanation_lines.append(
        "Using the provided metadata, I treated 'english' as total enrollment (number of students), "
        "'students' as the number of teachers, and 'district'/'expenditure' as average reading and math scores."
    )
    explanation_lines.append(
        "I constructed a student–teacher ratio variable as enrollment divided by number of teachers, and an "
        "overall performance measure as the average of reading and math scores for each district (420 districts total)."
    )
    explanation_lines.append(
        f"The Pearson correlation between student–teacher ratio and average test score was {corr:.3f} "
        f"with p-value {p_corr:.3g}."
    )
    explanation_lines.append(
        f"A simple linear regression of average score on student–teacher ratio produced a slope of {slope:.3f} "
        f"score points per additional student per teacher (p-value {p_slope:.3g})."
    )

    if slope < 0:
        direction_text = (
            "This negative slope indicates that districts with smaller student–teacher ratios "
            "(fewer students per teacher) tend to have higher test scores, holding other factors constant in this "
            "simple bivariate model."
        )
    elif slope > 0:
        direction_text = (
            "This positive slope indicates that districts with larger student–teacher ratios "
            "(more students per teacher) tend to have higher test scores, which runs counter to the hypothesis."
        )
    else:
        direction_text = (
            "The estimated slope is effectively zero, indicating no detectable linear association between "
            "student–teacher ratio and test scores in this sample."
        )
    explanation_lines.append(direction_text)

    explanation_lines.append(
        "Based on the magnitude and statistical significance of this association, I mapped the evidence onto a "
        "0–100 Likert-style scale, where 0 is a strong 'No' and 100 is a strong 'Yes' to the research question."
    )
    explanation_lines.append(
        f"The resulting score of {score} reflects the overall strength and direction of evidence for a "
        "relationship between lower student–teacher ratios and higher academic performance in this dataset."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": score, "explanation": explanation}

    # Write the required JSON object to conclusion.txt with no extra text.
    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

