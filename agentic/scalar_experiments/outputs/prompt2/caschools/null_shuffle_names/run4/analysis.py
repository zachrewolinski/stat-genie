import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata and data
    with open("info.json", "r") as f:
        info = json.load(f)

    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meaning using info.json descriptions
    # According to info.json:
    # - "english": total enrollment (students)
    # - "students": number of teachers
    # - "district": average reading score
    # - "expenditure": average math score
    # - "income": district average income
    # - "school": percent qualifying for CalWorks
    # - "computer": percent qualifying for reduced-price lunch
    # - "rownames": percent of English learners
    # - "grades": expenditure per student

    # Compute student–teacher ratio and test scores
    df = df.copy()
    # Guard against any zero-teacher entries
    df = df[df["students"] != 0].copy()

    df["stratio"] = df["english"] / df["students"]
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Basic association: correlations
    corr_testscr = float(df["stratio"].corr(df["testscr"]))
    corr_read = float(df["stratio"].corr(df["read_score"]))
    corr_math = float(df["stratio"].corr(df["math_score"]))

    # Simple linear regression: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    coef_str1 = float(model1.params["stratio"])
    pval_str1 = float(model1.pvalues["stratio"])

    # Multiple regression with key demographic and resource controls
    control_cols = ["income", "school", "computer", "rownames", "grades"]
    available_controls = [c for c in control_cols if c in df.columns]

    if available_controls:
        X2 = sm.add_constant(df[["stratio"] + available_controls])
        model2 = sm.OLS(df["testscr"], X2).fit()
        coef_str2 = float(model2.params["stratio"])
        pval_str2 = float(model2.pvalues["stratio"])
    else:
        model2 = None
        coef_str2 = np.nan
        pval_str2 = np.nan

    # Decide on conclusion: is a LOWER student–teacher ratio associated with HIGHER performance?
    # Interpretation:
    # - Negative coefficients/correlations mean: more students per teacher -> lower scores,
    #   so equivalently, lower ratios -> higher performance.
    # - We combine sign and significance from both simple and multiple regression.
    negative_simple = coef_str1 < 0
    significant_simple = pval_str1 < 0.05

    if model2 is not None:
        negative_mult = coef_str2 < 0
        significant_mult = pval_str2 < 0.05
    else:
        negative_mult = False
        significant_mult = False

    # Heuristic for response and confidence
    if negative_simple and significant_simple and (not model2 or negative_mult):
        response = "Yes"
        # Strong, consistent negative association
        confidence = 85
    elif negative_simple and (not model2 or negative_mult):
        # Negative association but not strongly significant
        response = "Yes"
        confidence = 65
    else:
        # No clear or opposite association
        response = "No"
        confidence = 70

    # Build explanation text with key numerical evidence
    explanation_parts = []

    explanation_parts.append(
        "I used the provided California school district data (420 K-6/K-8 districts) "
        "and the metadata in info.json to map the shuffled column names to their meanings. "
        "Total enrollment is stored in 'english' and the number of teachers in 'students', "
        "so I defined the student–teacher ratio as enrollment divided by teachers."
    )

    explanation_parts.append(
        "Average academic performance was measured using both reading and math scores, "
        "stored in the 'district' (reading) and 'expenditure' (math) columns; I averaged "
        "these into a composite test score."
    )

    explanation_parts.append(
        f"The correlation between the student–teacher ratio and the composite test score "
        f"is {corr_testscr:.3f} (reading: {corr_read:.3f}, math: {corr_math:.3f})."
    )

    explanation_parts.append(
        "A simple linear regression of the composite test score on the student–teacher "
        f"ratio yields a coefficient of {coef_str1:.3f} points per additional student per teacher "
        f"(p-value {pval_str1:.3f})."
    )

    if model2 is not None:
        explanation_parts.append(
            "To account for potential confounding, I also estimated a multiple regression "
            "including district income, poverty and lunch-program participation, English-learner share, "
            f"and per-student expenditure as controls. In this model, the coefficient on the "
            f"student–teacher ratio is {coef_str2:.3f} (p-value {pval_str2:.3f})."
        )

    if response == "Yes":
        explanation_parts.append(
            "Across these analyses, the association between the student–teacher ratio and test scores "
            "is generally negative, meaning that districts with fewer students per teacher tend to have "
            "higher average test performance, although the strength and statistical significance vary "
            "depending on the specification."
        )
    else:
        explanation_parts.append(
            "Taken together, these results do not show a consistently strong or statistically robust "
            "negative relationship between the student–teacher ratio and test scores; the evidence that "
            "smaller classes are associated with higher performance is weak in this dataset."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    # Write conclusion to conclusion.txt as a single JSON object
    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

