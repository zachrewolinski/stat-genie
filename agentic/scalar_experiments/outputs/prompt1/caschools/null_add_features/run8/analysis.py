import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and average test score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables used in the model.
    model_vars = ["score", "stratio", "calworks", "lunch", "income", "english"]
    model_df = df[model_vars].dropna()

    # Fit an OLS regression of achievement on student-teacher ratio
    # controlling for key socioeconomic covariates.
    formula = "score ~ stratio + calworks + lunch + income + english"
    model = smf.ols(formula, data=model_df).fit()

    coef = float(model.params["stratio"])
    p_value = float(model.pvalues["stratio"])

    # For additional context, compute the simple Pearson correlation as well.
    corr = float(np.corrcoef(model_df["stratio"], model_df["score"])[0, 1])

    # Decision rule: treat "associated with" as a statistically significant
    # (alpha = 0.05) negative relationship between ratio and achievement.
    alpha = 0.05
    associated = coef < 0 and p_value < alpha

    if associated:
        response = "Yes"
        explanation = (
            "I modeled average 5th-grade test performance (mean of reading and math "
            "scores) as a function of the student-teacher ratio and socioeconomic "
            "controls (CalWorks participation, reduced-price-lunch share, average "
            "district income, and share of English learners) using OLS on 420 "
            "California K-6/K-8 districts. The estimated coefficient on the "
            "student-teacher ratio is negative "
            f"({coef:.3f} points per additional student per teacher) and "
            f"statistically significant (p-value {p_value:.3f} < 0.05), with a "
            f"negative Pearson correlation of {corr:.3f} between the ratio and "
            "test scores. These results indicate that districts with lower "
            "student-teacher ratios tend to have higher average test scores, "
            "even after accounting for these demographic factors."
        )
    else:
        response = "No"
        explanation = (
            "I modeled average 5th-grade test performance (mean of reading and math "
            "scores) as a function of the student-teacher ratio and socioeconomic "
            "controls (CalWorks participation, reduced-price-lunch share, average "
            "district income, and share of English learners) using OLS on 420 "
            "California K-6/K-8 districts. The estimated coefficient on the "
            "student-teacher ratio is "
            f"{coef:.3f} points per additional student per teacher with a "
            f"p-value of {p_value:.3f}, and the simple Pearson correlation between "
            f"the ratio and test scores is {corr:.3f}. The effect size is small and "
            "not statistically significant at the 0.05 level, so this dataset does "
            "not provide strong evidence that lower student-teacher ratios are "
            "associated with higher average test scores once these demographic "
            "factors are taken into account."
        )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt with no extra text.
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
