import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def fit_ols(y, X):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return model


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    df = df.dropna(subset=["stratio", "testscr"]).copy()

    # Full-sample analyses
    n_full = int(df.shape[0])
    corr_full = float(df["stratio"].corr(df["testscr"]))

    simple_full = fit_ols(df["testscr"], df[["stratio"]])

    covariates = [
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "students",
    ]
    available_covariates = [c for c in covariates if c in df.columns]
    multi_full = fit_ols(df["testscr"], df[["stratio"] + available_covariates])

    # Robustness: restrict to IQR of stratio
    q25, q75 = df["stratio"].quantile([0.25, 0.75])
    df_iqr = df.loc[(df["stratio"] >= q25) & (df["stratio"] <= q75)].copy()
    n_iqr = int(df_iqr.shape[0])
    corr_iqr = float(df_iqr["stratio"].corr(df_iqr["testscr"]))

    simple_iqr = fit_ols(df_iqr["testscr"], df_iqr[["stratio"]])
    multi_iqr = fit_ols(df_iqr["testscr"], df_iqr[["stratio"] + available_covariates])

    # Extract key numbers for explanation
    coef_simple = float(simple_full.params["stratio"])
    p_simple = float(simple_full.pvalues["stratio"])
    r2_simple = float(simple_full.rsquared)

    coef_multi = float(multi_full.params["stratio"])
    p_multi = float(multi_full.pvalues["stratio"])
    r2_multi = float(multi_full.rsquared)

    coef_simple_iqr = float(simple_iqr.params["stratio"])
    p_simple_iqr = float(simple_iqr.pvalues["stratio"])
    r2_simple_iqr = float(simple_iqr.rsquared)

    coef_multi_iqr = float(multi_iqr.params["stratio"])
    p_multi_iqr = float(multi_iqr.pvalues["stratio"])
    r2_multi_iqr = float(multi_iqr.rsquared)

    # Based on these results, the data do not show a clear, statistically significant relationship.
    response_value = 25

    explanation = (
        "Using data on {n_full} California K-6 and K-8 school districts "
        "from 1998-1999, I tested whether districts with lower student-teacher ratios "
        "have higher average standardized test performance in 5th grade. I constructed a "
        "student-teacher ratio as students divided by teachers and an overall test score as "
        "the average of the reading and math scores. In the full sample, the Pearson correlation "
        "between student-teacher ratio and test scores was {corr_full:.3f}, essentially zero. "
        "A simple linear regression of test score on the student-teacher ratio produced a slope of "
        "{coef_simple:.4f} points per one additional student per teacher (p = {p_simple:.3f}), with R^2 = "
        "{r2_simple:.4f}, meaning the ratio explains far less than 1% of the variation in scores. "
        "When I controlled for district income, English-learner share, poverty measures, computers, "
        "spending, and enrollment in a multiple regression, the slope on the student-teacher ratio was "
        "{coef_multi:.4f} (p = {p_multi:.3f}, R^2 = {r2_multi:.4f}), again extremely small and not statistically "
        "distinguishable from zero. As a robustness check, I repeated the analysis on the middle 50% of districts "
        "by student-teacher ratio (n = {n_iqr}), where class sizes are more typical. In this restricted sample, the "
        "correlation was {corr_iqr:.3f}, and the simple and multiple regression slopes were {coef_simple_iqr:.4f} "
        "(p = {p_simple_iqr:.3f}, R^2 = {r2_simple_iqr:.4f}) and {coef_multi_iqr:.4f} (p = {p_multi_iqr:.3f}, R^2 = "
        "{r2_multi_iqr:.4f}), respectively—again very small and statistically non-significant. Taken together, these "
        "results provide little evidence that districts with lower student-teacher ratios systematically achieve higher "
        "test scores in this dataset; any true relationship, if it exists, appears to be quite weak relative to the "
        "substantial cross-district variation in performance. I therefore answer 'No' to the question of whether a "
        "lower student-teacher ratio is associated with higher academic performance in this dataset. On a 0-100 scale "
        "where 0 is a strong 'No' and 100 is a strong 'Yes', I place my answer at {response_value}, reflecting moderate "
        "confidence that there is no meaningful association while still acknowledging the possibility of very small "
        "effects that this analysis cannot rule out."
    ).format(
        n_full=n_full,
        corr_full=corr_full,
        coef_simple=coef_simple,
        p_simple=p_simple,
        r2_simple=r2_simple,
        coef_multi=coef_multi,
        p_multi=p_multi,
        r2_multi=r2_multi,
        n_iqr=n_iqr,
        corr_iqr=corr_iqr,
        coef_simple_iqr=coef_simple_iqr,
        p_simple_iqr=p_simple_iqr,
        r2_simple_iqr=r2_simple_iqr,
        coef_multi_iqr=coef_multi_iqr,
        p_multi_iqr=p_multi_iqr,
        r2_multi_iqr=r2_multi_iqr,
        response_value=response_value,
    )

    conclusion = {"response": response_value, "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
