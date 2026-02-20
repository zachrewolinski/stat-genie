import json

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    cols = [
        "str",
        "score",
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    df_model = df[cols].dropna()

    n = len(df_model)

    # Simple bivariate association
    r, p_corr = stats.pearsonr(df_model["str"], df_model["score"])

    x_simple = sm.add_constant(df_model["str"])
    model_simple = sm.OLS(df_model["score"], x_simple).fit()
    beta_str_simple = float(model_simple.params["str"])
    p_str_simple = float(model_simple.pvalues["str"])

    # Multivariable model with key covariates
    covariates = ["income", "english", "calworks", "lunch", "computer", "expenditure"]
    x_multi = sm.add_constant(df_model[["str"] + covariates])
    model_multi = sm.OLS(df_model["score"], x_multi).fit()
    beta_str_multi = float(model_multi.params["str"])
    p_str_multi = float(model_multi.pvalues["str"])
    ci_multi = model_multi.conf_int().loc["str"]
    ci_low_multi = float(ci_multi[0])
    ci_high_multi = float(ci_multi[1])

    # Effect size: predicted difference between low and high STR
    str_low = float(df_model["str"].quantile(0.1))
    str_high = float(df_model["str"].quantile(0.9))
    delta_str = str_high - str_low
    delta_score_multi = beta_str_multi * delta_str

    # Decide response based on direction and significance
    if (
        beta_str_simple < 0
        and beta_str_multi < 0
        and p_str_simple < 0.05
        and p_str_multi < 0.05
    ):
        response = "Yes"
    else:
        response = "No"

    # Confidence heuristic based on multivariable p-value and effect consistency
    if response == "Yes":
        if p_str_multi < 0.001:
            confidence = 95
        elif p_str_multi < 0.01:
            confidence = 90
        else:
            confidence = 80
        conclusion_sentence = (
            "Because the adjusted association is negative and statistically significant, "
            "I conclude that, within these observational data, lower student–teacher "
            "ratios are associated with higher academic performance, although the "
            "analysis cannot prove causality."
        )
    else:
        if p_str_multi > 0.2:
            confidence = 70
        else:
            confidence = 60
        conclusion_sentence = (
            "Because the adjusted association is very small and not statistically "
            "significant after controlling for observed district characteristics, I "
            "conclude that these data do not provide clear evidence that lower "
            "student–teacher ratios are associated with higher academic performance."
        )

    explanation = (
        "Using data from {n} California school districts, I defined the student–teacher "
        "ratio as students divided by teachers and an overall academic performance score "
        "as the average of reading and math test scores. The Pearson correlation between "
        "student–teacher ratio and performance was {r:.3f} (p = {p_corr:.3g}), indicating "
        "{direction_corr} association. In a simple linear regression of performance on "
        "student–teacher ratio, the slope was {beta_simple:.2f} points per additional "
        "student per teacher (p = {p_simple:.3g}). After adjusting for district income, "
        "English-learner share, CalWorks participation, reduced-price lunch share, "
        "computers per student, and expenditures per student, the regression coefficient "
        "for student–teacher ratio remained {beta_multi:.2f} with a 95% confidence "
        "interval of [{ci_low:.2f}, {ci_high:.2f}] and p = {p_multi:.3g}. Moving from the "
        "10th to the 90th percentile of the student–teacher ratio was associated with an "
        "estimated change of {delta_score:.1f} test-score points. {conclusion_sentence}"
    ).format(
        n=n,
        r=r,
        p_corr=p_corr,
        direction_corr="a negative" if r < 0 else "no clear"
        if abs(r) < 0.05
        else "a positive",
        beta_simple=beta_str_simple,
        p_simple=p_str_simple,
        beta_multi=beta_str_multi,
        ci_low=ci_low_multi,
        ci_high=ci_high_multi,
        p_multi=p_str_multi,
        delta_score=delta_score_multi,
        conclusion_sentence=conclusion_sentence,
    )

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
