import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata (not strictly required for the computation, but kept for context)
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    data_path = base_dir / "caschools.csv"
    df = pd.read_csv(data_path)

    # According to info.json descriptions:
    # - "english" is total enrollment
    # - "students" is the number of teachers (FTE)
    # Construct student-teacher ratio = students per teacher
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: use reading and math scores and their average
    read_col = "district"  # Average reading score
    math_col = "expenditure"  # Average math score
    df["avg_score"] = df[[read_col, math_col]].mean(axis=1)

    results = {}

    def run_simple_ols(y_col: str, x_col: str):
        y = df[y_col].astype(float)
        x = sm.add_constant(df[x_col].astype(float))
        model = sm.OLS(y, x, missing="drop").fit()
        return {
            "coef": float(model.params[x_col]),
            "p_value": float(model.pvalues[x_col]),
            "r_squared": float(model.rsquared),
        }

    results["avg"] = run_simple_ols("avg_score", "stratio")
    results["read"] = run_simple_ols(read_col, "stratio")
    results["math"] = run_simple_ols(math_col, "stratio")

    # Combine evidence: focus on the average score regression
    avg_res = results["avg"]
    coef = avg_res["coef"]
    p_value = avg_res["p_value"]
    r_squared = avg_res["r_squared"]

    # Determine strength on 0–100 Likert scale.
    # We treat 0 = strong "No", 50 = neutral, 100 = strong "Yes"
    # for the directional question "Is a lower student–teacher ratio associated with higher performance?".
    #
    # First compute a symmetric strength in [0, 50] based on significance and fit (association, not causation):
    # - Strong, highly significant association (p<0.001, R^2 >= 0.20) -> ~40
    # - Moderate association (p<0.01, R^2 ~0.10–0.20) -> ~30
    # - Weak but statistically significant (p<0.05, R^2 <0.10) -> ~20
    # - Little to no evidence (p>=0.05 or extremely small effect) -> ~10
    if p_value < 0.001 and r_squared >= 0.20:
        base_strength = 40.0
    elif p_value < 0.01 and r_squared >= 0.10:
        base_strength = 30.0
    elif p_value < 0.05:
        base_strength = 20.0
    else:
        base_strength = 10.0

    # Sign: the hypothesis is that *lower* ratio is associated with *higher* scores.
    # A negative coefficient of stratio on scores supports the hypothesis (Yes, >50);
    # a positive coefficient or essentially zero effect goes against it (No, <50).
    if coef < 0:
        response_score = 50.0 + base_strength
        direction_answer = "Yes"
    else:
        response_score = 50.0 - base_strength
        direction_answer = "No"

    # Clamp to [0, 100] and convert to int
    response_int = int(np.clip(round(response_score), 0, 100))

    # Build explanation string
    explanation_lines = []
    explanation_lines.append(
        f"Research question: {research_question}"
    )
    explanation_lines.append(
        "I constructed the student–teacher ratio as total enrollment divided by the number of teachers, "
        "using the variables described in the metadata (\"english\" for enrollment and \"students\" for teachers)."
    )
    explanation_lines.append(
        "I then defined academic performance as the average of the district-level reading and math scores "
        "(columns \"district\" and \"expenditure\")."
    )
    explanation_lines.append(
        "Using ordinary least squares regressions of reading, math, and their average on the student–teacher ratio, "
        "I examined the sign, magnitude, statistical significance, and R-squared of the association."
    )
    explanation_lines.append(
        f"For the regression of the average score on the student–teacher ratio, the estimated coefficient on the ratio "
        f"is {coef:.4f}, with p-value {p_value:.4g} and R-squared {r_squared:.3f}."
    )
    if coef < 0:
        explanation_lines.append(
            "The coefficient is negative, meaning that higher student–teacher ratios (more students per teacher) "
            "are associated with lower test scores, so lower ratios correspond to higher performance."
        )
    else:
        explanation_lines.append(
            "The coefficient is positive and very small, so in this sample higher student–teacher ratios are very weakly "
            "and non-significantly associated with *higher* scores, which goes against the hypothesized direction; "
            "combined with the near-zero R-squared, this provides little evidence for a meaningful relationship."
        )
    explanation_lines.append(
        "Based on the combination of statistical significance, effect size, and the direction of the coefficient, "
        "I translated the strength of evidence for the hypothesized negative association into a 0–100 Likert scale "
        "where 0 is a strong \"No\", 50 is neutral, and 100 is a strong \"Yes\"."
    )
    explanation_lines.append(
        f"The resulting score of {response_int} reflects a {direction_answer} answer: in this dataset there is "
        f"{'evidence for' if direction_answer == 'Yes' else 'little to no evidence for'} the claim that lower "
        "student–teacher ratios are associated with higher academic performance, and any observed association should "
        "be interpreted as observational rather than strictly causal."
    )

    explanation = " ".join(explanation_lines)

    # Write required JSON output
    conclusion = {"response": response_int, "explanation": explanation}
    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
