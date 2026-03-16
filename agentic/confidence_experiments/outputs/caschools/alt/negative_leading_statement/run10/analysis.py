import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "caschools.csv"

    with info_path.open() as f:
        info = json.load(f)

    research_questions = info.get("research_questions", [])
    question = research_questions[0] if research_questions else ""

    df = pd.read_csv(data_path)

    # Construct student-teacher ratio (students per teacher) and an overall test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation between student-teacher ratio and achievement.
    corr = df["stratio"].corr(df["testscr"])

    # Simple bivariate OLS: testscr ~ stratio.
    y = df["testscr"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    # Multivariate OLS controlling for key demographics and spending.
    control_candidates = ["english", "lunch", "calworks", "income", "expenditure"]
    controls = [c for c in control_candidates if c in df.columns]

    coef_multi = None
    p_multi = None
    model_multi = None

    if controls:
        X_multi = sm.add_constant(df[["stratio"] + controls])
        model_multi = sm.OLS(y, X_multi).fit()
        coef_multi = float(model_multi.params["stratio"])
        p_multi = float(model_multi.pvalues["stratio"])

    # Count how often we see a negative, statistically significant relationship.
    total_tests = 0
    negative_significant_count = 0

    if not np.isnan(coef_simple):
        total_tests += 1
        if coef_simple < 0 and p_simple < 0.05:
            negative_significant_count += 1

    if coef_multi is not None and not np.isnan(coef_multi):
        total_tests += 1
        if coef_multi < 0 and p_multi is not None and p_multi < 0.05:
            negative_significant_count += 1

    # Map evidence into a 0–100 Likert scale where 0 = strong "No" and 100 = strong "Yes"
    # to the question "Is a lower student-teacher ratio associated with higher academic performance?"
    if negative_significant_count == 0:
        response_value = 20  # Evidence does not support a clear association.
        conclusion_statement = (
            "Based on this dataset, I do not find clear evidence that lower student-teacher "
            "ratios are associated with higher academic performance."
        )
    else:
        # There is at least some statistically significant negative association.
        # If all estimated models agree and are strongly significant, treat as strong evidence.
        strong_simple = coef_simple < 0 and p_simple < 0.01
        strong_multi = (
            coef_multi is not None
            and coef_multi < 0
            and p_multi is not None
            and p_multi < 0.01
        )

        if total_tests > 0 and negative_significant_count == total_tests and (
            strong_simple or strong_multi
        ):
            response_value = 85
            conclusion_statement = (
                "There is strong evidence in this dataset that districts with smaller classes "
                "(fewer students per teacher) tend to have higher average test scores."
            )
        else:
            response_value = 70
            conclusion_statement = (
                "There is moderate but statistically reliable evidence that districts with smaller "
                "classes (fewer students per teacher) tend to have higher average test scores."
            )

    # Narrative explanation tying together the research question, correlation, and regressions.
    corr_text = (
        "The Pearson correlation between the student-teacher ratio (students per teacher) and "
        f"average test score is {corr:.3f}, indicating that districts with more students per "
        "teacher tend to have lower scores."
    )

    simple_text = (
        "A simple OLS regression of average test score on the student-teacher ratio yields a "
        f"coefficient of {coef_simple:.2f} with a p-value of {p_simple:.3g}, so an increase of "
        "one student per teacher is associated with a decrease in average test scores of roughly "
        f"{abs(coef_simple):.2f} points."
    )

    multi_text = ""
    if model_multi is not None:
        multi_text = (
            "An OLS regression that additionally controls for student demographics (poverty, "
            "English learners) and district spending yields a coefficient on the student-teacher "
            f"ratio of {coef_multi:.2f} with a p-value of {p_multi:.3g}, indicating that the "
            "negative association remains after adjusting for these covariates."
        )

    leading_text = (
        "The research question was posed with a prior belief that the answer is 'No', but the "
        "analysis here is based solely on the observed data using standard correlation and "
        "regression techniques."
    )

    pieces = [
        f"Research question: {question}",
        leading_text,
        corr_text,
        simple_text,
    ]

    if multi_text:
        pieces.append(multi_text)

    pieces.append(conclusion_statement)

    explanation = " ".join(pieces)

    # Ensure the response is an integer between 0 and 100.
    response_int = int(max(0, min(100, round(response_value))))

    result = {
        "response": response_int,
        "explanation": explanation,
    }

    with (base_dir / "conclusion.txt").open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

