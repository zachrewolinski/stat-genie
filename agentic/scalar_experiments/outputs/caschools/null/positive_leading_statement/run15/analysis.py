import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]  # student-teacher ratio
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used
    model_vars = ["testscr", "str", "income", "english", "lunch", "calworks"]
    df_model = df[model_vars].dropna()

    # Basic summaries
    n = int(len(df_model))
    str_mean = float(df_model["str"].mean())
    str_sd = float(df_model["str"].std())
    testscr_mean = float(df_model["testscr"].mean())
    testscr_sd = float(df_model["testscr"].std())

    # Correlation between student-teacher ratio and test score
    corr = float(df_model["str"].corr(df_model["testscr"]))

    # Simple bivariate OLS: testscr ~ str
    y = df_model["testscr"]
    X_simple = sm.add_constant(df_model["str"])
    model_simple = sm.OLS(y, X_simple).fit()

    coef_str_simple = float(model_simple.params["str"])
    pval_str_simple = float(model_simple.pvalues["str"])
    ci_lower_simple, ci_upper_simple = model_simple.conf_int().loc["str"].tolist()
    ci_lower_simple = float(ci_lower_simple)
    ci_upper_simple = float(ci_upper_simple)

    # Multivariate OLS controlling for observable covariates
    X_multi = df_model[["str", "income", "english", "lunch", "calworks"]]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(y, X_multi).fit()

    coef_str_multi = float(model_multi.params["str"])
    pval_str_multi = float(model_multi.pvalues["str"])
    ci_lower_multi, ci_upper_multi = model_multi.conf_int().loc["str"].tolist()
    ci_lower_multi = float(ci_lower_multi)
    ci_upper_multi = float(ci_upper_multi)

    # Decide on strength of evidence for a negative association
    # Criteria: sign consistently negative, p-value < 0.05 in both models,
    # and confidence intervals entirely below zero.
    strong_negative = (
        coef_str_simple < 0
        and coef_str_multi < 0
        and pval_str_simple < 0.05
        and pval_str_multi < 0.05
        and ci_upper_simple < 0
        and ci_upper_multi < 0
    )

    # Map evidence strength to Likert-style 0-100 response
    if strong_negative:
        response_score = 85
        overall_conclusion = (
            "Overall, the data provide fairly strong evidence that districts with lower "
            "student-teacher ratios tend to have higher academic performance."
        )
        yes_no_answer = "Yes"
    elif (coef_str_simple < 0 and pval_str_simple < 0.05) or (
        coef_str_multi < 0 and pval_str_multi < 0.05
    ):
        response_score = 65
        overall_conclusion = (
            "There is moderate but statistically significant evidence of a negative association "
            "between the student-teacher ratio and academic performance."
        )
        yes_no_answer = "Yes"
    elif coef_str_simple < 0 or coef_str_multi < 0:
        response_score = 45
        overall_conclusion = (
            "The estimated association between the student-teacher ratio and academic performance "
            "is negative but statistically weak, so the evidence for a relationship is limited."
        )
        yes_no_answer = "Unclear / weak Yes"
    else:
        response_score = 35
        overall_conclusion = (
            "The estimated association between the student-teacher ratio and academic performance "
            "is very small and not statistically significant, so the data do not support a clear "
            "relationship in the hypothesized direction."
        )
        yes_no_answer = "No"

    # Build explanation string
    explanation_parts = []
    explanation_parts.append(
        "Research question: Is a lower student-teacher ratio associated with higher academic performance "
        "in California K-6 and K-8 districts?"
    )
    explanation_parts.append(
        f"I constructed a student-teacher ratio variable as students/teachers (mean {str_mean:.2f}, "
        f"SD {str_sd:.2f}, N={n}) and an overall test score as the average of reading and math scores "
        f"(mean {testscr_mean:.2f}, SD {testscr_sd:.2f})."
    )
    if abs(corr) < 0.05:
        corr_interp = (
            "which is close to zero and indicates little linear relationship between the ratio and test scores."
        )
    elif corr < 0:
        corr_interp = (
            "which is negative and suggests that districts with smaller student-teacher ratios tend to have higher test scores."
        )
    else:
        corr_interp = (
            "which is positive and suggests that districts with larger student-teacher ratios tend to have slightly higher test scores."
        )

    explanation_parts.append(
        f"The Pearson correlation between student-teacher ratio and test scores is {corr:.3f}, {corr_interp}"
    )
    explanation_parts.append(
        "In a simple ordinary least squares regression of test scores on the student-teacher ratio, "
        f"the estimated coefficient on the ratio is {coef_str_simple:.3f} (95% CI [{ci_lower_simple:.3f}, "
        f"{ci_upper_simple:.3f}], p-value={pval_str_simple:.4f})."
    )
    explanation_parts.append(
        "I then estimated a multivariate regression adding controls for district income, percentages of "
        "students in CalWorks, qualifying for reduced-price lunch, and English learners. "
        f"In this model, the coefficient on the student-teacher ratio is {coef_str_multi:.3f} "
        f"(95% CI [{ci_lower_multi:.3f}, {ci_upper_multi:.3f}], p-value={pval_str_multi:.4f})."
    )
    explanation_parts.append(
        "However, these are observational cross-sectional data, so the analysis establishes an association "
        "rather than a causal effect; unmeasured confounding factors may still influence both class size and "
        "achievement."
    )
    explanation_parts.append(
        f"{overall_conclusion} Given this evidence, I answer '{yes_no_answer}' to the research question and "
        f"encode this on the 0-100 Likert scale as {response_score}."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
