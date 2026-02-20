import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def compute_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived variables needed for the analysis.

    Based on the metadata in info.json, the column semantics are:
    - english: total enrollment (number of students)
    - students: number of teachers
    - district: average reading score
    - expenditure: average math score
    """
    df = df.copy()

    # Student-teacher ratio: students per teacher.
    # Guard against division by zero or missing values.
    df["student_teacher_ratio"] = df["english"] / df["students"]

    # Overall academic performance: average of reading and math scores.
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop obviously invalid ratios (e.g., non-finite values)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["student_teacher_ratio", "avg_score"]
    )

    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    """
    Assess whether a lower student–teacher ratio is associated
    with higher academic performance.
    """
    ratio = df["student_teacher_ratio"]
    score = df["avg_score"]

    # Simple Pearson correlation
    corr = ratio.corr(score)

    # Linear regression of score on ratio
    X = sm.add_constant(ratio)
    model = sm.OLS(score, X).fit()
    slope = model.params["student_teacher_ratio"]
    p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    # Decide on binary answer based on sign and significance
    # Lower ratio => expect negative slope (higher score when ratio is smaller).
    is_negative_association = slope < 0
    is_statistically_significant = p_value < 0.05

    if is_negative_association and is_statistically_significant:
        response = "Yes"
        # Strength informed by effect size (corr, R^2) and p-value.
        # Map |corr| (~0–1) into 0–70, significance adds up to 30.
        base_strength = min(abs(corr), 1.0) * 70
        sig_bonus = 30
        strength = int(round(min(base_strength + sig_bonus, 100)))
    else:
        response = "No"
        # If slope is small or positive, strength reflects lack of
        # strong, consistent evidence for the hypothesized relationship.
        # Use inverse of |corr| to capture "absence" of strong pattern.
        base_strength = (1.0 - min(abs(corr), 1.0)) * 60
        # Penalize if there is any hint of a pattern (smaller p)
        significance_penalty = max(0.0, (0.05 - min(p_value, 0.05)) / 0.05) * 20
        strength = int(round(min(base_strength + significance_penalty, 100)))

    # Confidence: primarily based on sample size and model fit.
    n = df.shape[0]
    # Sample-size contribution: up to 60 points around n=400.
    size_component = min(n / 400.0, 1.0) * 60
    # Model-fit contribution: up to 40 points based on R^2.
    fit_component = min(max(r_squared, 0.0), 1.0) * 40
    confidence = int(round(min(size_component + fit_component, 100)))

    explanation_lines = [
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?",
        "Constructed student-teacher ratio as total enrollment divided by number of teachers, using the metadata from info.json.",
        "Defined academic performance as the average of district-level reading and math scores.",
        f"Across {n} districts, the correlation between student-teacher ratio and average test score was {corr:.3f}.",
        f"In a linear regression of average test score on student-teacher ratio, the estimated slope was {slope:.3f} points per unit increase in the ratio (p-value = {p_value:.4g}, R-squared = {r_squared:.3f}).",
    ]

    if response == "Yes":
        explanation_lines.append(
            "The negative and statistically significant slope indicates that districts with lower student-teacher ratios tend to have higher test scores, even after accounting for sampling variability."
        )
    else:
        if not is_negative_association:
            explanation_lines.append(
                "The estimated slope is not negative, so the data do not support the hypothesis that lower ratios are systematically associated with higher performance."
            )
        if not is_statistically_significant:
            explanation_lines.append(
                "The slope is not statistically different from zero at the 5% level, suggesting any apparent relationship could be due to random variation."
            )
        explanation_lines.append(
            "Overall, the evidence does not show a strong or reliable association between lower student-teacher ratios and higher academic performance in this dataset."
        )

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def write_conclusion(conclusion: dict, path: Path) -> None:
    # Ensure the JSON object is the only content in the file.
    with path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def main() -> None:
    csv_path = Path("caschools.csv")
    df_raw = load_data(csv_path)
    df = compute_variables(df_raw)
    conclusion = analyze_relationship(df)
    write_conclusion(conclusion, Path("conclusion.txt"))


if __name__ == "__main__":
    main()

