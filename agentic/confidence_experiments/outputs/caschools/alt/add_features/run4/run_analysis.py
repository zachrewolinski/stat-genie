import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata and research question
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_questions = info.get("research_questions", [])
    research_question = research_questions[0] if research_questions else ""

    # Load data
    data_path = base_dir / "caschools.csv"
    df = pd.read_csv(data_path)

    # Construct key analytic variables
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables (should be none, but for safety)
    df_model = df[["testscr", "stratio"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Bivariate OLS: testscr ~ stratio
    X = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model = sm.OLS(y, X).fit()

    coef_stratio = model.params["stratio"]
    pvalue_stratio = model.pvalues["stratio"]
    r2 = model.rsquared

    # Direction: negative coefficient means higher scores at lower student-teacher ratios.
    association_is_positive = coef_stratio < 0 and pvalue_stratio < 0.05

    # Map statistical evidence to Likert scale (0–100)
    if pvalue_stratio < 0.001:
        base_response = 90
    elif pvalue_stratio < 0.01:
        base_response = 80
    elif pvalue_stratio < 0.05:
        base_response = 70
    elif pvalue_stratio < 0.1:
        base_response = 55
    else:
        base_response = 40

    if not association_is_positive:
        # If the estimated relationship does not support "lower ratio -> higher scores",
        # reflect that by flipping around a neutral midpoint of 50.
        response = max(0, min(100, int(round(100 - base_response))))
    else:
        response = max(0, min(100, int(round(base_response))))

    # Build explanation text
    direction_text = (
        "a negative and statistically significant"
        if coef_stratio < 0 and pvalue_stratio < 0.05
        else "not a statistically significant"
        if pvalue_stratio >= 0.05
        else "a weakly significant"
    )

    explanation = (
        f"Research question: {research_question}\n\n"
        "I used the California K-6 and K-8 district dataset (420 districts) and "
        "constructed the student–teacher ratio as students divided by full-time equivalent teachers, "
        "and an overall academic performance measure as the average of 5th grade reading and math scores. "
        "I then fit a simple ordinary least squares regression of average test score on the student–teacher ratio.\n\n"
        f"The estimated coefficient on the student–teacher ratio was {coef_stratio:.3f}, with a p-value of "
        f"{pvalue_stratio:.3g} and an R-squared of {r2:.3f}. This indicates {direction_text} relationship "
        "between the student–teacher ratio and average test scores: as the ratio increases (more students per teacher), "
        "average scores tend to change by the estimated coefficient amount per one additional student per teacher. "
        "A negative significant coefficient implies that lower student–teacher ratios are associated with higher academic performance.\n\n"
        f"Based on the sign and statistical significance of this coefficient (p-value {pvalue_stratio:.3g}) and the "
        "explained variance (R-squared), I conclude that there is "
        f"{'clear' if association_is_positive and pvalue_stratio < 0.01 else 'some' if association_is_positive else 'insufficient'} "
        "evidence in this dataset that districts with lower student–teacher ratios have higher academic performance. "
        f"The Likert-scale response of {response} reflects the strength of this evidence, where values closer to 100 represent "
        "a stronger \"Yes\" answer to the research question."
    )

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)


if __name__ == "__main__":
    main()

