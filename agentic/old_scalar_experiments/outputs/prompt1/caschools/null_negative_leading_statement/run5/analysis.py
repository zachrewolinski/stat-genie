import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Define student-teacher ratio (students per teacher) and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used in the models
    covariates = ["income", "calworks", "lunch", "english", "expenditure", "computer"]
    model_vars = ["stratio", "testscr"] + covariates
    df_model = df[model_vars].dropna()

    # Simple correlations
    corr_testscr = df_model["stratio"].corr(df_model["testscr"])
    corr_read = df_model["stratio"].corr(df["read"].loc[df_model.index])
    corr_math = df_model["stratio"].corr(df["math"].loc[df_model.index])

    corr_direction = "negative" if corr_testscr < 0 else "positive"

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    coef_s_simple = float(model_simple.params["stratio"])
    p_s_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    if coef_s_simple < 0:
        effect_simple = f"about {-coef_s_simple:.2f} points lower"
    else:
        effect_simple = f"about {coef_s_simple:.2f} points higher"

    # Multiple regression with key covariates
    X_full = sm.add_constant(df_model[["stratio"] + covariates])
    model_full = sm.OLS(y, X_full).fit()

    coef_s = float(model_full.params["stratio"])
    p_s = float(model_full.pvalues["stratio"])
    ci_s_low, ci_s_high = model_full.conf_int().loc["stratio"]
    ci_s_low = float(ci_s_low)
    ci_s_high = float(ci_s_high)
    r2_full = float(model_full.rsquared)

    # Decide whether there is evidence that lower student-teacher ratio
    # is associated with higher academic performance.
    associated = coef_s < 0 and p_s < 0.05
    response = "Yes" if associated else "No"

    if associated:
        conclusion_sentence = (
            "Because across districts a higher student-teacher ratio is consistently "
            "associated with lower academic performance, even after adjusting for "
            "socioeconomic and resource covariates, the data support the conclusion "
            "that lower student-teacher ratios are associated with higher academic "
            "performance in this dataset."
        )
    else:
        conclusion_sentence = (
            "After adjusting for socioeconomic and resource covariates, the estimated "
            "association between student-teacher ratio and academic performance is "
            "weak or statistically inconclusive, so this dataset does not provide "
            "clear evidence that lower student-teacher ratios are associated with "
            "higher academic performance."
        )

    explanation = (
        "We used data on 420 California K-6 and K-8 school districts, defining the "
        "student-teacher ratio as total enrollment divided by the number of teachers "
        "and academic performance as the average of district-level fifth-grade reading "
        "and math scores. "
        f"The simple correlation between the student-teacher ratio and overall test "
        f"scores was {corr_testscr:.3f} ({corr_direction} association), while the "
        f"correlations with reading and math scores separately were {corr_read:.3f} "
        f"and {corr_math:.3f}, respectively. "
        f"In a simple linear regression of overall test scores on the student-teacher "
        f"ratio, the coefficient on the ratio was {coef_s_simple:.3f} (p = "
        f"{p_s_simple:.3g}, R² = {r2_simple:.3f}), meaning that each additional student "
        f"per teacher was associated with {effect_simple} average test score. "
        "To account for potential confounding, we fit a multiple regression including "
        "district income, the percentage of students on income assistance (CalWorks), "
        "the percentage qualifying for reduced-price lunch, the percentage of English "
        "learners, per-pupil expenditure, and the number of computers. "
        f"In this adjusted model, the coefficient on the student-teacher ratio was "
        f"{coef_s:.3f} with a p-value of {p_s:.3g}, a 95% confidence interval from "
        f"{ci_s_low:.3f} to {ci_s_high:.3f}, and an R² of {r2_full:.3f}. "
        + conclusion_sentence
    )

    result = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

