import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_metadata(info_path: Path) -> str:
    with info_path.open() as f:
        info = json.load(f)
    questions = info.get("research_questions") or []
    return questions[0] if questions else ""


def analyze_relationship(df: pd.DataFrame) -> dict:
    # Compute student-teacher ratio as students per teacher.
    data = df[["feature6", "feature7", "feature14", "feature15"]].dropna().copy()
    data["student_teacher_ratio"] = data["feature6"] / data["feature7"]
    data["avg_score"] = data[["feature14", "feature15"]].mean(axis=1)

    x = data["student_teacher_ratio"].to_numpy()
    y = data["avg_score"].to_numpy()

    # Correlation between ratio and performance.
    r, p_corr = stats.pearsonr(x, y)

    # Simple linear regression: avg_score ~ student_teacher_ratio.
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = float(model.params[1])
    p_slope = float(model.pvalues[1])
    r_squared = float(model.rsquared)

    return {
        "n": int(data.shape[0]),
        "correlation": float(r),
        "corr_p_value": float(p_corr),
        "slope": slope,
        "slope_p_value": p_slope,
        "r_squared": r_squared,
    }


def score_likert(result: dict) -> int:
    r = result["correlation"]
    p = result["slope_p_value"]
    slope = result["slope"]

    supports_hypothesis = slope < 0  # higher ratio (more students/teacher) lowers scores
    magnitude = abs(r)

    if np.isnan(r) or np.isnan(p) or result["n"] == 0:
        return 50

    if p >= 0.05:
        # Insufficient evidence for a relationship.
        return 45 if supports_hypothesis else 30

    # Statistically significant relationship.
    if supports_hypothesis:
        if magnitude < 0.1:
            return 60
        if magnitude < 0.3:
            return 75
        if magnitude < 0.5:
            return 85
        return 95
    else:
        # Significant relationship in the opposite direction.
        if magnitude < 0.1:
            return 40
        if magnitude < 0.3:
            return 25
        if magnitude < 0.5:
            return 15
        return 5


def build_explanation(question: str, result: dict, score: int) -> str:
    direction = "negative" if result["slope"] < 0 else "positive"
    supports_text = (
        "supports the idea that lower student-teacher ratios are associated with higher academic performance"
        if result["slope"] < 0 and result["slope_p_value"] < 0.05
        else "does not provide strong evidence that lower student-teacher ratios are associated with higher academic performance"
    )

    explanation = (
        f"Research question: {question}\n"
        "Using the provided California school districts data, I computed the student-teacher ratio "
        "as total enrollment divided by the number of teachers (students per teacher) and defined academic performance "
        "as the average of the district's average reading and math scores.\n"
        f"There are {result['n']} districts with complete data. The Pearson correlation between the student-teacher ratio "
        f"and average test score is {result['correlation']:.3f} (p = {result['corr_p_value']:.4g}), indicating a {direction} association.\n"
        f"A simple linear regression of average test score on the student-teacher ratio yields a slope of {result['slope']:.3f} "
        f"score points per additional student per teacher (p = {result['slope_p_value']:.4g}, R^2 = {result['r_squared']:.3f}). This {supports_text}.\n"
        f"Based on the sign, statistical significance, and magnitude of the relationship, I place my answer at {score} on a 0–100 scale, "
        "where 0 is a strong 'No' and 100 is a strong 'Yes' to the existence of an association between lower student-teacher ratios and higher academic performance."
    )
    return explanation


def main() -> None:
    base = Path(__file__).resolve().parent
    info_path = base / "info.json"
    data_path = base / "caschools.csv"
    output_path = base / "conclusion.txt"

    question = load_metadata(info_path)
    df = pd.read_csv(data_path)

    result = analyze_relationship(df)
    score = score_likert(result)
    explanation = build_explanation(question, result, score)

    conclusion = {"response": int(score), "explanation": explanation}

    with output_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

