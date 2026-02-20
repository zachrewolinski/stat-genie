import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent
    data_path = base_dir / "caschools.csv"
    info_path = base_dir / "info.json"
    output_path = base_dir / "conclusion.txt"

    # Load metadata (not strictly needed for computation, but used for context)
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    df = pd.read_csv(data_path)

    # Construct variables based on metadata:
    # feature6: total enrollment (students)
    # feature7: number of teachers
    # feature14: average reading score
    # feature15: average math score
    df = df.copy()
    df["students"] = df["feature6"]
    df["teachers"] = df["feature7"]

    # Student–teacher ratio as students per teacher (lower is better)
    df["student_teacher_ratio"] = df["students"] / df["teachers"]

    df["reading_score"] = df["feature14"]
    df["math_score"] = df["feature15"]
    df["avg_score"] = df[["reading_score", "math_score"]].mean(axis=1)

    # Basic association: correlation
    corr = df["avg_score"].corr(df["student_teacher_ratio"])

    # Simple linear regression of avg_score on student_teacher_ratio
    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(df["avg_score"], X).fit()

    slope = float(model.params["student_teacher_ratio"])
    p_value = float(model.pvalues["student_teacher_ratio"])
    r_squared = float(model.rsquared)

    # Map statistical evidence to a 0–100 Likert-style response
    if slope < 0 and p_value < 0.001 and abs(corr) > 0.2:
        response = 85
        strength_desc = "strong"
    elif slope < 0 and p_value < 0.01:
        response = 75
        strength_desc = "moderately strong"
    elif slope < 0 and p_value < 0.05:
        response = 65
        strength_desc = "modest"
    elif slope < 0:
        response = 55
        strength_desc = "weak"
    elif slope >= 0 and p_value < 0.05:
        response = 20
        strength_desc = "evidence against"
    else:
        response = 50
        strength_desc = "little clear"

    explanation_lines = [
        f"Research question: {research_question}",
        "Data: 420 California K–6 and K–8 districts with 5th grade Stanford 9 test scores and district-level characteristics.",
        "Student–teacher ratio was computed as total enrollment divided by the number of teachers (students per teacher), so a lower ratio means fewer students per teacher.",
        "Academic performance was summarized as the average of the district reading and math scores.",
        f"The Pearson correlation between average test score and the student–teacher ratio is {corr:.3f}, indicating that districts with more students per teacher tend to have "
        f"{'lower' if corr < 0 else 'higher' if corr > 0 else 'similar'} test scores.",
        f"A simple OLS regression of average test score on the student–teacher ratio yields a slope of {slope:.3f} "
        f"(points of test score per additional student per teacher), with p-value {p_value:.3g} and R-squared {r_squared:.3f}.",
        "This means that, holding nothing else constant, moving to larger classes (higher student–teacher ratios) is statistically associated with "
        f"{'lower' if slope < 0 else 'higher' if slope > 0 else 'no clear change in'} achievement in this dataset, "
        "although the proportion of variance explained is relatively small.",
        f"Given the direction of the association (slope sign), its statistical significance (p-value), and the modest explanatory power (R-squared), "
        f"I judge there is {strength_desc} evidence that lower student–teacher ratios are associated with higher academic performance in this dataset.",
        f"On a 0–100 scale, where 0 is a strong 'No' and 100 is a strong 'Yes', I assign a score of {response}.",
    ]

    explanation = "\n".join(explanation_lines)

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with output_path.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

