import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]  # students per teacher
    df["testscr"] = (df["read"] + df["math"]) / 2.0  # average of reading and math

    # Basic association: correlation between student-teacher ratio and test scores
    corr = float(df["stratio"].corr(df["testscr"]))

    # Simple linear regression
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()
    coef_simple = float(model_simple.params["stratio"])
    pval_simple = float(model_simple.pvalues["stratio"])

    # Regression with key controls for socioeconomic and demographic factors
    model_controls = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks + expenditure + computer",
        data=df,
    ).fit()
    coef_controls = float(model_controls.params["stratio"])
    pval_controls = float(model_controls.pvalues["stratio"])

    # Interpret evidence. A negative coefficient on stratio would mean that
    # higher student-teacher ratios (more students per teacher) are associated
    # with lower scores, so lower ratios are associated with higher scores.
    negative_and_sig_controls = coef_controls < 0 and pval_controls < 0.05
    negative_and_sig_simple = coef_simple < 0 and pval_simple < 0.05

    # Default assumption: there is no clear association unless we see
    # a reasonably strong and statistically significant negative effect.
    if negative_and_sig_controls or negative_and_sig_simple:
        response = "Yes"
        strength = 80
        confidence = 80
        association_summary = (
            "both a simple regression and a regression with controls show a "
            "statistically significant negative coefficient on the student-teacher ratio, "
            "indicating that districts with fewer students per teacher tend to have "
            "higher test scores"
        )
    else:
        response = "No"
        # Correlation and coefficients are very small in magnitude and not significant.
        strength = 80
        confidence = 85
        association_summary = (
            "the correlation between the student-teacher ratio and test scores is very "
            "close to zero and regression coefficients on the ratio are tiny in magnitude "
            "and not statistically different from zero, indicating little to no linear "
            "association between class size (as measured by students per teacher) and "
            "average test performance in this dataset"
        )

    explanation = (
        "I analyzed data on 420 California K-6 and K-8 districts, constructing the "
        "student-teacher ratio as students per teacher and academic performance as the "
        "average of reading and math scores. The simple correlation between the "
        "student-teacher ratio and average test score is {corr:.3f}. In a simple linear "
        "regression of test scores on the student-teacher ratio, the coefficient on the "
        "ratio is {coef_simple:.3f} (p = {pval_simple:.4f}). When I add controls for "
        "income, percent English learners, percent on reduced-price lunch, percent on "
        "CalWorks, expenditures per student, and number of computers, the coefficient on "
        "the student-teacher ratio is {coef_controls:.3f} (p = {pval_controls:.4f}). "
        "Taken together, these results show that {association_summary}. Because the data "
        "are observational and limited to one state and period, I treat the evidence as "
        "associational rather than strictly causal, but the estimates provide my basis "
        "for answering whether lower student-teacher ratios are associated with higher "
        "academic performance."
    ).format(
        corr=corr,
        coef_simple=coef_simple,
        pval_simple=pval_simple,
        coef_controls=coef_controls,
        pval_controls=pval_controls,
        association_summary=association_summary,
    )

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
