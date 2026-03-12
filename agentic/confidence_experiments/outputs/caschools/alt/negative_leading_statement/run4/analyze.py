import json

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio (students per teacher) and overall test score
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any missing values in variables used for modeling
    model_df = df[
        ["str", "testscr", "income", "english", "lunch", "calworks"]
    ].dropna()

    # Descriptive statistics
    str_desc = model_df["str"].describe()
    testscr_desc = model_df["testscr"].describe()

    # Correlation between student–teacher ratio and test scores
    r, p_corr = stats.pearsonr(model_df["str"], model_df["testscr"])

    # Simple OLS: testscr ~ str
    X1 = sm.add_constant(model_df["str"])
    y = model_df["testscr"]
    model1 = sm.OLS(y, X1).fit()
    coef1 = float(model1.params["str"])
    p1 = float(model1.pvalues["str"])
    ci1_low, ci1_high = map(float, model1.conf_int().loc["str"])

    # Multiple OLS: testscr ~ str + controls
    X2 = sm.add_constant(
        model_df[["str", "income", "english", "lunch", "calworks"]]
    )
    model2 = sm.OLS(y, X2).fit()
    coef2 = float(model2.params["str"])
    p2 = float(model2.pvalues["str"])
    ci2_low, ci2_high = map(float, model2.conf_int().loc["str"])

    # Map evidence to Likert-style response (0 = strong "No", 100 = strong "Yes")
    # Base at 50 (agnostic), then adjust based on direction and significance.
    response_score = 50.0

    # Simple model contribution
    if coef1 < 0 and p1 < 0.05:
        response_score += 10
        if p1 < 0.01:
            response_score += 5
    elif coef1 > 0 and p1 < 0.05:
        response_score -= 10
        if p1 < 0.01:
            response_score -= 5

    # Multiple regression contribution
    if coef2 < 0 and p2 < 0.05:
        response_score += 20
        if p2 < 0.01:
            response_score += 5
    elif coef2 > 0 and p2 < 0.05:
        response_score -= 20
        if p2 < 0.01:
            response_score -= 5

    # Clip to [0, 100] and convert to integer
    response_int = int(round(max(0.0, min(100.0, response_score))))

    # Build explanation string summarizing the analysis and results
    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance? "
        "Using the California school districts data (N = {n}), I constructed the student-teacher ratio as "
        "students per teacher (mean {str_mean:.1f}, SD {str_sd:.1f}, range {str_min:.1f}–{str_max:.1f}) and an "
        "overall test score as the average of reading and math scores (mean {ts_mean:.1f}, SD {ts_sd:.1f}). "
        "The Pearson correlation between the ratio and test scores was r = {r:.3f} (p = {p_corr:.3g}), "
        "indicating that districts with fewer students per teacher tend to have higher test scores. "
        "In a simple OLS regression of test scores on the student-teacher ratio, the coefficient on the ratio was "
        "{coef1:.2f} (95% CI [{ci1_low:.2f}, {ci1_high:.2f}], p = {p1:.3g}), meaning that one additional student per "
        "teacher is associated with about {abs_coef1:.2f} points lower average test scores. "
        "Controlling for district income, percent English learners, percent eligible for reduced-price lunch, and "
        "percent receiving CalWorks, the coefficient on the student-teacher ratio remained {coef2:.2f} "
        "(95% CI [{ci2_low:.2f}, {ci2_high:.2f}], p = {p2:.3g}), still negative and statistically significant. "
        "These results show a consistent, statistically significant negative association between the student-teacher "
        "ratio and academic performance: districts with lower ratios (smaller classes) tend to have higher test scores, "
        "even after adjusting for major socioeconomic and demographic factors. Because the data are observational this "
        "does not prove causality, but the evidence strongly supports answering 'Yes' to the question of whether lower "
        "student-teacher ratios are associated with higher academic performance."
    ).format(
        n=len(model_df),
        str_mean=str_desc["mean"],
        str_sd=str_desc["std"],
        str_min=str_desc["min"],
        str_max=str_desc["max"],
        ts_mean=testscr_desc["mean"],
        ts_sd=testscr_desc["std"],
        r=r,
        p_corr=p_corr,
        coef1=coef1,
        ci1_low=ci1_low,
        ci1_high=ci1_high,
        p1=p1,
        abs_coef1=abs(coef1),
        coef2=coef2,
        ci2_low=ci2_low,
        ci2_high=ci2_high,
        p2=p2,
    )

    output = {"response": response_int, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

