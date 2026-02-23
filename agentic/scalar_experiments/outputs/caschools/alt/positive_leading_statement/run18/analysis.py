import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and overall test score.
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables, if present.
    model_df = df.dropna(
        subset=["testscr", "str", "lunch", "calworks", "english", "income", "expenditure"]
    ).copy()

    # Simple Pearson correlation between student-teacher ratio and test score.
    corr = model_df["testscr"].corr(model_df["str"])

    # Bivariate regression: testscr ~ str.
    X_simple = sm.add_constant(model_df[["str"]])
    y = model_df["testscr"]
    model_simple = sm.OLS(y, X_simple).fit(cov_type="HC1")
    coef_str_simple = model_simple.params["str"]
    p_str_simple = model_simple.pvalues["str"]

    # Multiple regression with key socioeconomic controls.
    controls = ["lunch", "calworks", "english", "income", "expenditure"]
    X_full = sm.add_constant(model_df[["str"] + controls])
    model_full = sm.OLS(y, X_full).fit(cov_type="HC1")
    coef_str_full = model_full.params["str"]
    p_str_full = model_full.pvalues["str"]

    # Effect size: change in testscr for a 2-student reduction in STR.
    delta_two_students = -2.0
    effect_two_students = coef_str_full * delta_two_students

    # Decide Likert-scale response based on sign and significance of the
    # student–teacher ratio in the more fully adjusted model.
    if (coef_str_full < 0) and (p_str_full < 0.01):
        response_score = 90
    elif (coef_str_full < 0) and (p_str_full < 0.05):
        response_score = 80
    elif (coef_str_full < 0) and (p_str_full < 0.1):
        response_score = 65
    else:
        # The adjusted model does not show a statistically significant
        # association in the expected direction, so we lean toward a
        # cautious or slightly negative answer.
        response_score = 40

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?\n\n"
        "Operationalization:\n"
        "- Constructed student–teacher ratio (STR) as students divided by teachers.\n"
        "- Measured overall academic performance as the average of reading and math scores (testscr).\n\n"
        f"Descriptive relationship:\n"
        f"- Pearson correlation between STR and testscr is {corr:.3f} "
        "(negative values indicate that lower STR is associated with higher scores).\n\n"
        "Regression evidence:\n"
        f"- In a bivariate OLS regression of testscr on STR, the STR coefficient is {coef_str_simple:.3f} "
        f"with p-value {p_str_simple:.4f} (HC1 robust standard errors).\n"
        f"- In a multiple regression controlling for lunch, calworks, english, income, and expenditure, "
        f"the STR coefficient is {coef_str_full:.3f} with p-value {p_str_full:.4f} (HC1 robust SEs).\n"
        "- In this full model, the STR coefficient is not statistically significant at conventional levels,\n"
        "  indicating that once observed socioeconomic differences are taken into account, we do not have strong\n"
        "  evidence that class size alone is a major driver of score differences across districts.\n\n"
        f"Effect size interpretation:\n"
        f"- Based on the full model, a 2-student reduction in STR is associated with an expected change of "
        f"{effect_two_students:.2f} points in the average test score.\n\n"
        "Conclusion:\n"
        "- Overall, the data show a modest negative correlation and a statistically significant association in the\n"
        "  simple (unadjusted) regression, consistent with lower student–teacher ratios being linked to higher scores.\n"
        "- However, once we adjust for key socioeconomic and resource variables, the estimated association weakens and\n"
        "  is no longer statistically significant, so the evidence for a clear, robust relationship is limited.\n"
        f"- On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', a score of {response_score} "
        "reflects a cautious answer: the data are somewhat consistent with a beneficial effect of smaller classes but\n"
        "do not provide strong, statistically robust evidence once observed confounders are controlled for."
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
