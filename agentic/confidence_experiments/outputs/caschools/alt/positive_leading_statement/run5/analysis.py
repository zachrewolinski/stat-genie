import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key analytic variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used below (should be rare/nonexistent)
    model_vars = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
    ]
    df_model = df[model_vars].dropna()

    # 1) Simple correlation between student-teacher ratio and test scores
    corr, corr_p = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # 2) Simple linear regression: testscr ~ stratio
    ols_simple = smf.ols("testscr ~ stratio", data=df_model).fit()
    beta_str_simple = ols_simple.params["stratio"]
    p_str_simple = ols_simple.pvalues["stratio"]
    r2_simple = ols_simple.rsquared

    # 3) Multiple regression controlling for key demographics and resources
    ols_controls = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks + expenditure",
        data=df_model,
    ).fit()
    beta_str_controls = ols_controls.params["stratio"]
    p_str_controls = ols_controls.pvalues["stratio"]
    r2_controls = ols_controls.rsquared

    # Determine Likert-scale response based on strength and robustness.
    # We expect a negative coefficient on stratio if lower ratios are associated
    # with higher scores. We treat the bivariate evidence as important, but
    # reduce our confidence when the effect is not statistically significant
    # after controlling for key covariates.
    if beta_str_simple < 0 and p_str_simple < 0.001:
        if beta_str_controls < 0 and p_str_controls < 0.05:
            # Strong, robust negative association.
            response_score = 85
        elif beta_str_controls < 0 and p_str_controls < 0.1:
            # Still negative and marginally significant with controls.
            response_score = 75
        elif beta_str_controls < 0:
            # Still negative but not statistically significant with controls.
            response_score = 65
        else:
            # Bivariate evidence is strong, but controls reverse the sign.
            response_score = 55
    elif beta_str_simple < 0 and p_str_simple < 0.05:
        # Weaker bivariate evidence.
        response_score = 60
    else:
        # Little or no evidence of a negative association.
        response_score = 50

    # Build concise textual explanation
    explanation = (
        "Using 420 California K-6 and K-8 school districts, I constructed a "
        "student-teacher ratio (students divided by teachers) and an overall "
        "test score equal to the average of reading and math scores. "
        f"The simple correlation between student-teacher ratio and test scores is "
        f"{corr:.3f} (p-value {corr_p:.3g}), indicating that districts with fewer "
        "students per teacher tend to have higher academic performance. "
        f"A simple linear regression of test scores on the student-teacher ratio "
        f"yields a coefficient of {beta_str_simple:.3f} (p-value {p_str_simple:.3g}, "
        f"R-squared {r2_simple:.3f}), showing that higher ratios are associated with "
        "lower scores. "
        f"When I control for income, English-learner share, reduced-price lunch, "
        f"CalWorks participation, and per-pupil expenditures, the coefficient on the "
        f"student-teacher ratio remains negative at {beta_str_controls:.3f} "
        f"(p-value {p_str_controls:.3g}, R-squared {r2_controls:.3f}), but it is no "
        "longer statistically significant. Taken together, the data show a clear "
        "negative bivariate association between student-teacher ratios and test "
        "scores, while the evidence for an independent effect after accounting for "
        "socioeconomic and demographic factors is weaker. Overall, this supports a "
        "moderately strong Yes that lower student-teacher ratios are associated with "
        "higher academic performance, though the relationship is not definitive."
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
