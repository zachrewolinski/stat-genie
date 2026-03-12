import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Reconstruct key variables based on metadata descriptions in info.json.
    # Column semantics (true meaning -> column name):
    # - Total enrollment -> "english"
    # - Number of teachers -> "students"
    # - Average reading score -> "district"
    # - Average math score -> "expenditure"
    # Student–teacher ratio (STR) = enrollment / number of teachers
    df = df.copy()
    df["enrollment"] = df["english"]
    df["num_teachers"] = df["students"]

    # Guard against division by zero
    df = df[df["num_teachers"] > 0].copy()

    df["stratio"] = df["enrollment"] / df["num_teachers"]

    # Academic performance proxy: average of reading and math scores
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Drop rows with missing values in variables of interest
    df = df.dropna(subset=["stratio", "testscr"])

    n_obs = int(df.shape[0])

    # Simple correlation between STR and test scores
    corr = float(df["stratio"].corr(df["testscr"]))

    # Simple bivariate regression: testscr ~ stratio
    X1 = sm.add_constant(df[["stratio"]])
    model1 = sm.OLS(df["testscr"], X1).fit()

    beta1 = float(model1.params["stratio"])
    se1 = float(model1.bse["stratio"])
    p1 = float(model1.pvalues["stratio"])

    # Multiple regression with common controls to check robustness:
    # income (district income), rownames (percent English learners),
    # school (percent in CalWorks), computer (percent reduced-price lunch),
    # grades (expenditure per student).
    controls = ["income", "rownames", "school", "computer", "grades"]
    available_controls = [c for c in controls if c in df.columns]
    X2 = sm.add_constant(df[["stratio"] + available_controls])
    model2 = sm.OLS(df["testscr"], X2).fit()

    beta2 = float(model2.params["stratio"])
    se2 = float(model2.bse["stratio"])
    p2 = float(model2.pvalues["stratio"])

    # Build a Likert-scale response (0–100) reflecting both direction and strength
    # of the evidence that lower STR (smaller classes) is associated with higher scores.
    score = 50.0

    # Direction of association (we expect a negative coefficient: higher STR -> lower scores)
    if beta1 < 0:
        score += 10
    else:
        score -= 10

    if beta2 < 0:
        score += 5
    else:
        score -= 5

    # Statistical significance from both models
    for p_val in (p1, p2):
        if p_val < 0.001:
            score += 15
        elif p_val < 0.01:
            score += 10
        elif p_val < 0.05:
            score += 5
        elif p_val < 0.1:
            score += 0
        else:
            score -= 5

    # Strength via correlation magnitude
    abs_r = abs(corr)
    if abs_r > 0.5:
        score += 10
    elif abs_r > 0.3:
        score += 5
    elif abs_r < 0.1:
        score -= 5

    # Clip to [0, 100] and convert to int
    score_int = int(np.clip(round(score), 0, 100))

    # Build human-readable explanation
    explanation_lines = []
    explanation_lines.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?"
    )
    explanation_lines.append(
        f"Sample and construction: Using {n_obs} California K-6/K-8 districts, "
        "I reconstructed the student-teacher ratio as total enrollment divided by the number of teachers, "
        "and defined academic performance as the average of district-level reading and math scores."
    )
    explanation_lines.append(
        f"Simple association: The correlation between the student-teacher ratio and average test score is {corr:.3f}, "
        "indicating that districts with smaller classes (lower ratios) tend to have higher scores when this number is negative."
    )
    explanation_lines.append(
        "Bivariate regression (testscr ~ STR): "
        f"the estimated coefficient on the student-teacher ratio is {beta1:.3f} "
        f"(SE = {se1:.3f}, p-value = {p1:.4f}). "
        "A negative and statistically significant coefficient implies that, on average, increasing the ratio "
        "is associated with lower academic performance."
    )
    explanation_lines.append(
        "Multiple regression with controls: Adding income, percent English learners, percent in CalWorks, "
        "percent on reduced-price lunch, and per-pupil expenditure yields a coefficient on the student-teacher ratio of "
        f"{beta2:.3f} (SE = {se2:.3f}, p-value = {p2:.4f}). "
        "This checks whether the association persists after adjusting for key demographic and resource differences across districts."
    )

    if beta1 < 0 and beta2 < 0 and p1 < 0.05 and p2 < 0.05:
        qualitative = (
            "Across both specifications, the student-teacher ratio remains negative and statistically significant, "
            "indicating consistent evidence that smaller classes are associated with higher academic performance."
        )
    elif (beta1 < 0 and p1 < 0.05) or (beta2 < 0 and p2 < 0.05):
        qualitative = (
            "At least one specification shows a negative and statistically significant relationship between the student-teacher ratio "
            "and test scores, providing moderate evidence that smaller classes are associated with higher academic performance, "
            "though the strength of the association is sensitive to model choice."
        )
    else:
        qualitative = (
            "The estimated coefficients do not consistently show a negative and statistically significant relationship, "
            "so the data provide at most weak evidence that smaller classes are associated with higher academic performance."
        )

    explanation_lines.append(qualitative)
    explanation_lines.append(
        f"On a 0–100 Likert scale where 0 means a strong 'No' and 100 means a strong 'Yes' to the existence of a relationship, "
        f"I summarize the overall evidence with a response value of {score_int}. "
        "Values above 50 indicate evidence in favor of a negative association (smaller classes, higher performance), "
        "with higher values reflecting stronger and more robust statistical support."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": score_int, "explanation": explanation}

    # Write required JSON to conclusion.txt with no extra text
    output_path = Path("conclusion.txt")
    output_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

