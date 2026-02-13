import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used
    model_vars = [
        "avg_score",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
    ]
    df_model = df[model_vars].dropna().copy()

    # Simple correlation between student-teacher ratio and academic performance
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["avg_score"])

    # Simple linear regression
    model_simple = smf.ols("avg_score ~ stratio", data=df_model).fit()
    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    ci_simple_low, ci_simple_high = model_simple.conf_int().loc["stratio"]

    # Multiple regression with key socioeconomic controls
    formula_controls = (
        "avg_score ~ stratio + income + english + lunch + calworks + expenditure"
    )
    model_ctrl = smf.ols(formula_controls, data=df_model).fit()
    coef_ctrl = model_ctrl.params["stratio"]
    p_ctrl = model_ctrl.pvalues["stratio"]
    ci_ctrl_low, ci_ctrl_high = model_ctrl.conf_int().loc["stratio"]

    # Decide answer based on direction and significance of association
    associated = (coef_ctrl < 0) and (p_ctrl < 0.05)
    response = "Yes" if associated else "No"

    # Summaries for explanation
    n = int(df_model.shape[0])
    mean_str = df_model["stratio"].mean()
    sd_str = df_model["stratio"].std()
    mean_score = df_model["avg_score"].mean()
    sd_score = df_model["avg_score"].std()

    # Interpretation text that reflects the actual estimates
    if abs(r) < 0.05 or p_corr >= 0.05:
        corr_phrase = (
            "showing essentially no linear relationship between student–teacher ratio "
            "and average test scores."
        )
    elif r < 0:
        corr_phrase = (
            "indicating that districts with fewer students per teacher tend to have "
            "higher scores."
        )
    else:
        corr_phrase = (
            "indicating that districts with more students per teacher tend to have "
            "higher scores."
        )

    if associated:
        inference_phrase = (
            "Because the adjusted association is negative and statistically "
            "significant at the 5% level, I conclude that in this dataset, lower "
            "student–teacher ratios are associated with higher academic performance, "
            "recognizing that the analysis is observational and cannot by itself "
            "prove causality."
        )
    else:
        inference_phrase = (
            "Because the adjusted coefficient on student–teacher ratio is small in "
            "magnitude and not statistically distinguishable from zero at conventional "
            "levels, the data do not provide strong evidence that lower student–teacher "
            "ratios are associated with higher academic performance; the results are "
            "consistent with little or no relationship in this sample."
        )

    explanation = (
        "Using data on {n} California K-6/K-8 school districts, "
        "I constructed the student–teacher ratio as students divided by teachers "
        "(mean {mean_str:.1f}, SD {sd_str:.1f}) and academic performance as the "
        "average of reading and math scores (mean {mean_score:.1f}, SD {sd_score:.1f}). "
        "The simple Pearson correlation between student–teacher ratio and average test "
        "score is r = {r:.3f} (p = {p_corr:.3g}), {corr_phrase} "
        "In a simple linear regression of average score on student–teacher ratio, "
        "a one-student increase in the ratio is associated with a change of "
        "{coef_simple:.2f} points in average score "
        "(95% CI [{ci_simple_low:.2f}, {ci_simple_high:.2f}], p = {p_simple:.3g}). "
        "After adjusting for district income, percentage of English learners, "
        "percent of students on income assistance, percent on reduced-price lunch, "
        "and per-pupil expenditure, the coefficient on student–teacher ratio is "
        "{coef_ctrl:.2f} (95% CI [{ci_ctrl_low:.2f}, {ci_ctrl_high:.2f}], "
        "p = {p_ctrl:.3g}). "
        "{inference_phrase}"
    ).format(
        n=n,
        mean_str=mean_str,
        sd_str=sd_str,
        mean_score=mean_score,
        sd_score=sd_score,
        r=r,
        p_corr=p_corr,
        coef_simple=coef_simple,
        ci_simple_low=ci_simple_low,
        ci_simple_high=ci_simple_high,
        p_simple=p_simple,
        coef_ctrl=coef_ctrl,
        ci_ctrl_low=ci_ctrl_low,
        ci_ctrl_high=ci_ctrl_high,
        p_ctrl=p_ctrl,
        corr_phrase=corr_phrase,
        inference_phrase=inference_phrase,
    )

    # Ensure single-line JSON by replacing newlines with spaces
    explanation_clean = " ".join(explanation.split())

    conclusion = {
        "response": response,
        "explanation": explanation_clean,
    }

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
