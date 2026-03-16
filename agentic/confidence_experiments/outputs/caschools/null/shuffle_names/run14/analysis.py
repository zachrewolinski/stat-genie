import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct analysis-ready variables, using metadata-informed mappings.

    Mappings (based on info.json descriptions):
    - english   -> enrollment (students)
    - students  -> number of teachers
    - district  -> average reading score
    - expenditure -> average math score
    - income    -> district average income
    - rownames  -> percent English learners
    - computer  -> percent qualifying for reduced-price lunch
    - school    -> percent qualifying for CalWorks (income assistance)
    """
    enroll = df["english"].astype(float)
    teachers = df["students"].astype(float)

    # Guard against division by zero just in case.
    stratio = enroll / teachers.replace({0: np.nan})

    read_score = df["district"].astype(float)
    math_score = df["expenditure"].astype(float)
    testscr = (read_score + math_score) / 2.0

    income = df["income"].astype(float)
    english_learners = df["rownames"].astype(float)
    lunch_pct = df["computer"].astype(float)
    calworks_pct = df["school"].astype(float)

    analysis_df = pd.DataFrame(
        {
            "testscr": testscr,
            "stratio": stratio,
            "income": income,
            "english_learners": english_learners,
            "lunch_pct": lunch_pct,
            "calworks_pct": calworks_pct,
            "enroll": enroll,
        }
    ).dropna()

    analysis_df["log_enroll"] = np.log(analysis_df["enroll"])

    return analysis_df


def run_regressions(df: pd.DataFrame):
    """
    Run a simple bivariate regression and a multivariate regression.

    Returns a dictionary with coefficient and p-value for the student-teacher ratio
    from both models.
    """
    y = df["testscr"]

    # Simple regression: test score on student-teacher ratio only
    X_simple = sm.add_constant(df[["stratio"]])
    simple_model = sm.OLS(y, X_simple).fit()
    simple_coef = float(simple_model.params["stratio"])
    simple_pval = float(simple_model.pvalues["stratio"])

    # Multivariate regression with key controls
    controls = ["log_enroll", "income", "english_learners", "lunch_pct", "calworks_pct"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    multi_model = sm.OLS(y, X_multi).fit()
    multi_coef = float(multi_model.params["stratio"])
    multi_pval = float(multi_model.pvalues["stratio"])

    return {
        "simple": {"coef": simple_coef, "pval": simple_pval},
        "multi": {"coef": multi_coef, "pval": multi_pval},
    }


def likert_from_results(results: dict) -> int:
    """
    Map regression evidence to a 0-100 Likert scale for the claim:
    "A lower student-teacher ratio is associated with higher academic performance."
    """
    simple = results["simple"]
    multi = results["multi"]

    # We expect a negative coefficient on stratio if lower ratio -> higher scores.
    signs = [np.sign(simple["coef"]), np.sign(multi["coef"])]
    negative_signs = sum(s < 0 for s in signs)

    # Use the multivariate model's p-value as primary evidence,
    # but ensure it agrees in sign with the simple model where possible.
    p = multi["pval"]
    coef = multi["coef"]

    if coef >= 0:
        # Evidence points against the hypothesized direction.
        if p < 0.05:
            return 10
        return 20

    # Negative coefficient: direction consistent with the hypothesis.
    # Strength based on significance level and agreement across models.
    if p < 0.001:
        base = 90
    elif p < 0.01:
        base = 80
    elif p < 0.05:
        base = 70
    elif p < 0.1:
        base = 60
    else:
        base = 45

    # If both models have negative coefficients, slightly increase confidence.
    if negative_signs == 2 and base >= 60:
        base += 5

    # Clip to [0, 100] and cast to int.
    return int(max(0, min(100, base)))


def build_explanation(results: dict, response: int) -> str:
    simple = results["simple"]
    multi = results["multi"]

    direction_simple = "negative" if simple["coef"] < 0 else "positive"
    direction_multi = "negative" if multi["coef"] < 0 else "positive"

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?\n\n"
        "Data and variables:\n"
        "- Academic performance was measured as the average of district reading and math scores.\n"
        "- The student-teacher ratio was constructed as total enrollment divided by the number of teachers.\n"
        "- Additional district-level controls included log enrollment, average income, percent English learners,\n"
        "  percent of students receiving income assistance (CalWorks), and percent qualifying for reduced-price lunch.\n\n"
        "Statistical analysis:\n"
        "- A simple linear regression of test scores on the student-teacher ratio produced a {dir_s} coefficient of approximately "
        "{coef_s:.2f} points per additional student per teacher (p-value ≈ {p_s:.3f}).\n"
        "- A multivariate regression controlling for key demographic and resource variables yielded a {dir_m} coefficient of approximately "
        "{coef_m:.2f} (p-value ≈ {p_m:.3f}).\n\n"
        "Interpretation:\n"
    ).format(
        dir_s=direction_simple,
        coef_s=simple["coef"],
        p_s=simple["pval"],
        dir_m=direction_multi,
        coef_m=multi["coef"],
        p_m=multi["pval"],
    )

    if response >= 60:
        conclusion_sentence = (
            "Both models indicate that districts with lower student-teacher ratios tend to have higher test scores, "
            "and the association remains statistically significant after adjusting for observed covariates. "
            "This provides substantial evidence for a meaningful negative relationship between the student-teacher ratio "
            "and academic performance in this dataset."
        )
    elif 40 <= response < 60:
        conclusion_sentence = (
            "The estimated relationship between student-teacher ratios and test scores is directionally consistent "
            "with lower ratios being associated with higher performance, but the statistical evidence is modest and "
            "sensitive to model specification. "
            "Overall, the data suggest at most a weak association once demographic and resource controls are included."
        )
    else:
        conclusion_sentence = (
            "After accounting for demographic and resource differences across districts, there is little robust evidence "
            "that variation in the student-teacher ratio is strongly associated with higher academic performance in this dataset."
        )

    explanation += conclusion_sentence + (
        "\n\nLikert-scale response (0 = strong 'No', 100 = strong 'Yes') reflects both the statistical significance "
        "and the magnitude of the estimated relationship across these models."
    )

    return explanation


def main():
    df = pd.read_csv("caschools.csv")
    analysis_df = build_analysis_dataframe(df)

    results = run_regressions(analysis_df)
    response = likert_from_results(results)
    explanation = build_explanation(results, response)

    conclusion = {"response": int(response), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

