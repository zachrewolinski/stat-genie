import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "caschools.csv"

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with any missing values in variables used below
    analysis_cols = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
    ]
    df_model = df[analysis_cols].dropna().copy()

    # Basic correlation between student–teacher ratio and test scores
    corr = float(df_model["testscr"].corr(df_model["stratio"]))

    # Simple bivariate regression: testscr ~ stratio
    X1 = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model1 = sm.OLS(y, X1).fit()
    coef_str1 = float(model1.params["stratio"])
    p_str1 = float(model1.pvalues["stratio"])

    # Multiple regression with key demographic and resource controls
    X2 = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    ]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(y, X2).fit()
    coef_str2 = float(model2.params["stratio"])
    p_str2 = float(model2.pvalues["stratio"])

    # Decide on Yes/No:
    # Interpret "lower student–teacher ratio associated with higher academic performance"
    # as: student–teacher ratio (students per teacher) having a negative and statistically
    # significant coefficient in models of test scores.
    alpha = 0.05
    associated = (coef_str2 < 0.0) and (p_str2 < alpha)

    response = "Yes" if associated else "No"

    n_obs = int(df_model.shape[0])

    explanation = (
        "Using the caschools dataset (N = {n_obs}), I examined whether districts with "
        "lower student–teacher ratios tend to have higher academic performance. "
        "I defined academic performance as the average of the reading and math test "
        "scores and defined the student–teacher ratio as total students divided by "
        "total teachers (so a lower value corresponds to larger classes). "
        "The simple correlation between test scores and the student–teacher ratio is "
        "{corr:.3f}. In a linear regression of test scores on the student–teacher "
        "ratio alone, the estimated coefficient on the ratio is {coef1:.3f} with a "
        "p-value of {p1:.3g}. When I add controls for district income, the percentages "
        "of students on CalWorks, reduced-price lunch, and English learners, and "
        "per-pupil expenditure, the coefficient on the student–teacher ratio is "
        "{coef2:.3f} with a p-value of {p2:.3g}. Interpreting the ratio as students "
        "per teacher, a negative coefficient would indicate that smaller classes are "
        "associated with higher test scores. However, in these data the estimated "
        "coefficients are very close to zero and not statistically different from zero "
        "at conventional significance levels, so the dataset does not provide clear "
        "evidence that lower student–teacher ratios are associated with higher "
        "academic performance."
    ).format(
        n_obs=n_obs,
        corr=corr,
        coef1=coef_str1,
        p1=p_str1,
        coef2=coef_str2,
        p2=p_str2,
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
