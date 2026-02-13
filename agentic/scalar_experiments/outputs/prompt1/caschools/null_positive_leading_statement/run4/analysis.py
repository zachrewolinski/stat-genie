import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    cols = ["stratio", "testscr", "income", "english", "lunch", "calworks"]
    df_clean = df[cols].dropna()

    # Basic correlation
    corr, corr_p = stats.pearsonr(df_clean["stratio"], df_clean["testscr"])

    # Simple linear regression: testscr ~ stratio
    X1 = sm.add_constant(df_clean["stratio"])
    model1 = sm.OLS(df_clean["testscr"], X1).fit()
    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])

    # Multiple regression with controls
    X2 = sm.add_constant(df_clean[["stratio", "income", "english", "lunch", "calworks"]])
    model2 = sm.OLS(df_clean["testscr"], X2).fit()
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])

    # Decide on overall answer based on direction and significance
    associated = (coef1 < 0 and pval1 < 0.05) or (coef2 < 0 and pval2 < 0.05)
    response = "Yes" if associated else "No"

    # Build human-readable explanation
    direction_simple = "decrease" if coef1 < 0 else "increase"
    direction_control = "negative" if coef2 < 0 else "positive"

    explanation = (
        f"Using data on {len(df_clean)} California K-6 and K-8 school districts, "
        f"I computed the student-teacher ratio as students divided by teachers and "
        f"average academic performance as the mean of reading and math scores. "
        f"The Pearson correlation between the student-teacher ratio and average test score "
        f"was {corr:.3f} (p-value {corr_p:.3g}), indicating that districts with lower "
        f"student-teacher ratios tend to have higher scores when the correlation is negative. "
        f"In a simple linear regression of average test score on the student-teacher ratio, "
        f"each additional student per teacher was associated with a {abs(coef1):.2f}-point "
        f"{direction_simple} in the average test score (p-value {pval1:.3g}). "
        f"After controlling for district income, the percentage of English learners, the "
        f"percentage of students on reduced-price lunch, and the percentage receiving CalWorks, "
        f"the coefficient on the student-teacher ratio remained {direction_control} at "
        f"{coef2:.2f} (p-value {pval2:.3g}). "
        f"Taken together, these results {'support' if associated else 'do not support'} the conclusion "
        f"that lower student-teacher ratios are associated with higher academic performance."
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

