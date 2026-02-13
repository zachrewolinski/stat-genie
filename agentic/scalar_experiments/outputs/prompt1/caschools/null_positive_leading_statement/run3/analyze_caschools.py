import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used for modeling (if any)
    model_vars_simple = ["avg_score", "student_teacher_ratio"]
    model_vars_full = model_vars_simple + [
        "income",
        "lunch",
        "calworks",
        "english",
        "computer",
        "expenditure",
    ]

    df_simple = df[model_vars_simple].dropna()
    df_full = df[model_vars_full].dropna()

    # Simple bivariate regression: avg_score ~ student_teacher_ratio
    X_simple = sm.add_constant(df_simple["student_teacher_ratio"])
    y_simple = df_simple["avg_score"]
    model_simple = sm.OLS(y_simple, X_simple).fit()
    beta_str_simple = model_simple.params["student_teacher_ratio"]
    p_str_simple = model_simple.pvalues["student_teacher_ratio"]

    # Multivariate regression controlling for key demographics/resources
    X_full = sm.add_constant(
        df_full[
            [
                "student_teacher_ratio",
                "income",
                "lunch",
                "calworks",
                "english",
                "computer",
                "expenditure",
            ]
        ]
    )
    y_full = df_full["avg_score"]
    model_full = sm.OLS(y_full, X_full).fit()
    beta_str_full = model_full.params["student_teacher_ratio"]
    p_str_full = model_full.pvalues["student_teacher_ratio"]

    # Decision rule:
    # Treat the statement as supported ("Yes") if the estimated effect
    # of the student–teacher ratio is negative and statistically significant
    # at the 5% level in the multivariate model (our primary specification).
    supported = (beta_str_full < 0) and (p_str_full < 0.05)
    response = "Yes" if supported else "No"

    direction_simple = (
        "negative (higher ratios are associated with lower scores)"
        if beta_str_simple < 0
        else "positive (higher ratios are associated with higher scores)"
    )
    direction_full = (
        "negative (higher ratios are associated with lower scores)"
        if beta_str_full < 0
        else "positive (higher ratios are associated with higher scores)"
    )

    if supported:
        conclusion_sentence = (
            "Because the preferred multivariate specification shows a statistically significant negative "
            "association between the student–teacher ratio and average test scores, I conclude that the data "
            "provide evidence that lower student–teacher ratios are associated with higher academic performance."
        )
    else:
        conclusion_sentence = (
            "Because the preferred multivariate specification does not show a statistically significant "
            "association between the student–teacher ratio and average test scores, I conclude that these data "
            "do not provide strong evidence that lower student–teacher ratios are associated with higher "
            "academic performance."
        )

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance? "
        "Using data on 420 California school districts, I constructed a student–teacher ratio as total "
        "enrollment divided by the number of teachers and an overall performance measure as the average of "
        "5th-grade reading and math scores.\n\n"
        "I first fit a simple linear regression of average test score on the student–teacher ratio alone. "
        f"In this bivariate model, the estimated coefficient on the student–teacher ratio was {beta_str_simple:.3f} "
        f"with a p-value of {p_str_simple:.3g}. This indicates a {direction_simple} in the simple association.\n\n"
        "To account for important confounders, I then fit a multivariate regression including district income, "
        "the percentages of students on income assistance (CalWorks), eligible for reduced-price lunch, and who "
        "are English learners, as well as the number of computers and expenditure per student. "
        f"In this full model, the coefficient on the student–teacher ratio was {beta_str_full:.3f} "
        f"with a p-value of {p_str_full:.3g}. This corresponds to a {direction_full} adjusted association.\n\n"
        f"{conclusion_sentence} This conclusion is based on the observed direction, magnitude, and statistical "
        "significance of the estimated effect, rather than on any prior belief about the answer."
    )

    conclusion = {"response": response, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
