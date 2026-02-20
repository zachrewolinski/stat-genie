import json
from typing import Dict

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_analysis() -> Dict[str, object]:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Keep only rows with complete data on main controls
    cols = ["score", "stratio", "income", "english", "lunch", "calworks"]
    df_model = df[cols].dropna()

    n = len(df_model)

    # Simple correlation
    corr = df_model["score"].corr(df_model["stratio"])

    # Simple regression: score on student-teacher ratio
    simple_model = smf.ols("score ~ stratio", data=df_model).fit()
    simple_coef = float(simple_model.params["stratio"])
    simple_p = float(simple_model.pvalues["stratio"])
    simple_r2 = float(simple_model.rsquared)

    # Multiple regression with key demographic controls
    multi_model = smf.ols(
        "score ~ stratio + income + english + lunch + calworks",
        data=df_model,
    ).fit()
    multi_coef = float(multi_model.params["stratio"])
    multi_p = float(multi_model.pvalues["stratio"])
    multi_r2 = float(multi_model.rsquared)

    # Determine direction and statistical support for the research question:
    # \"Is a lower student–teacher ratio associated with higher academic performance?\"
    # That corresponds to a negative association between the ratio (students per teacher)
    # and test scores.
    has_negative_association = corr < 0 and simple_coef < 0 and multi_coef < 0
    statistically_significant = simple_p < 0.05 and multi_p < 0.05

    response = "Yes" if has_negative_association and statistically_significant else "No"

    # Map evidence to strength and confidence scores (0–100)
    base_strength = min(100, max(0, int(round(abs(corr) * 100))))
    if statistically_significant and has_negative_association:
        base_strength = min(100, base_strength + 15)

    if statistically_significant and has_negative_association:
        if simple_p < 0.001 and multi_p < 0.001:
            confidence = 95
        elif simple_p < 0.01 and multi_p < 0.01:
            confidence = 90
        else:
            confidence = 80
    else:
        confidence = 60

    strength = base_strength

    # Build explanation that reflects the actual sign and significance
    if corr < 0:
        corr_direction_text = (
            "indicating that districts with more students per teacher (larger ratios) "
            "tend to have lower test scores (a negative association)."
        )
    elif corr > 0:
        corr_direction_text = (
            "indicating that districts with more students per teacher (larger ratios) "
            "tend to have slightly higher test scores (a positive association), although "
            "the magnitude is very small."
        )
    else:
        corr_direction_text = (
            "indicating essentially no linear association between the student–teacher "
            "ratio and test scores."
        )

    if simple_p < 0.05:
        simple_sig_text = (
            "This slope is statistically different from zero at conventional "
            "significance levels."
        )
    else:
        simple_sig_text = (
            "This slope is not statistically different from zero at conventional "
            "significance levels, so the data do not show a clear linear relationship."
        )

    if multi_p < 0.05:
        multi_sig_text = (
            "The coefficient remains statistically significant after adding these "
            "controls."
        )
    else:
        multi_sig_text = (
            "The coefficient is not statistically significant after adding these "
            "controls, indicating that once we adjust for demographics and income, "
            "there is still no clear linear relationship."
        )

    if has_negative_association and statistically_significant:
        overall_text = (
            "Overall, the combination of a negative correlation and negative, "
            "statistically significant regression coefficients suggests that lower "
            "student–teacher ratios are associated with higher academic performance in "
            "this dataset. However, because the data are observational and aggregated "
            "at the district level, the results speak to association rather than a "
            "guaranteed causal effect of reducing class size."
        )
    else:
        overall_text = (
            "Taken together, the near-zero correlation and regression coefficients that "
            "are small in magnitude and not statistically significant suggest that, in "
            "this dataset, there is not strong evidence that lower student–teacher "
            "ratios are associated with higher academic performance. Any true effect, "
            "if it exists, is likely modest relative to other district-level factors "
            "such as demographics and income, and cannot be distinguished clearly from "
            "noise with these data."
        )

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "(measured by the average of reading and math scores) across California K–6/K–8 school districts.\n\n"
        f"The analysis uses {n} districts with complete data. I constructed the student–teacher ratio as "
        "'students / teachers' and the academic performance measure as the average of reading and math test "
        "scores.\n\n"
        f"First, I examined the simple Pearson correlation between the student–teacher ratio and average "
        f"test score. The correlation is {corr:.3f}, {corr_direction_text}\n\n"
        f"Second, I estimated a simple linear regression of average test score on the student–teacher ratio. "
        f"The estimated coefficient on the ratio is {simple_coef:.3f}, with a p-value of {simple_p:.4f} and an "
        f"R-squared of {simple_r2:.3f}. {simple_sig_text}\n\n"
        f"Third, I estimated a multiple regression that adds key demographic and socioeconomic controls "
        f"(family income, percentage of English learners, percentage on reduced-price lunch, and percentage "
        f"on public assistance). In this model, the coefficient on the student–teacher ratio is {multi_coef:.3f} "
        f"with a p-value of {multi_p:.4f} and an R-squared of {multi_r2:.3f}. {multi_sig_text}\n\n"
        f"{overall_text}"
    )

    return {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }


def main() -> None:
    result = run_analysis()
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
