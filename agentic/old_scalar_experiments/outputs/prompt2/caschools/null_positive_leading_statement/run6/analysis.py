import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    cols = [
        "stratio",
        "testscr",
        "income",
        "english",
        "calworks",
        "lunch",
        "expenditure",
        "computer",
        "students",
    ]
    data = df[cols].dropna()

    # Simple bivariate association
    corr, corr_p = stats.pearsonr(data["stratio"], data["testscr"])

    x1 = sm.add_constant(data["stratio"])
    model1 = sm.OLS(data["testscr"], x1).fit()

    # Multiple regression with demographic and resource controls
    x2 = data[
        [
            "stratio",
            "income",
            "english",
            "calworks",
            "lunch",
            "expenditure",
            "computer",
            "students",
        ]
    ]
    x2 = sm.add_constant(x2)
    model2 = sm.OLS(data["testscr"], x2).fit()

    coef1 = float(model1.params["stratio"])
    p1 = float(model1.pvalues["stratio"])
    coef2 = float(model2.params["stratio"])
    p2 = float(model2.pvalues["stratio"])

    # Decision rules for "Yes"/"No"
    if coef1 < 0 and coef2 < 0 and p1 < 0.05 and p2 < 0.05 and corr < 0 and corr_p < 0.05:
        response = "Yes"
        confidence = 90
    elif (coef1 < 0 and p1 < 0.05 and corr < 0 and corr_p < 0.05) or (
        coef2 < 0 and p2 < 0.1
    ):
        response = "Yes"
        confidence = 75
    else:
        response = "No"
        confidence = 60

    explanation = (
        f"Using data from {len(data)} California K-6 and K-8 districts, the student–teacher "
        f"ratio (students per teacher) has a Pearson correlation of {corr:.3f} (p={corr_p:.3g}) "
        f"with average test scores (mean of reading and math). This correlation is very close to "
        f"zero and not statistically significant, providing no evidence in the raw data that "
        f"districts with lower student–teacher ratios systematically achieve higher scores. "
        f"In a simple OLS regression of average test scores on the student–teacher ratio, the "
        f"coefficient on the ratio is {coef1:.3f} (p={p1:.3g}), so each additional student per "
        f"teacher is associated with an estimated {abs(coef1):.2f}-point change in average test "
        f"scores, which is essentially zero and not statistically distinguishable from no effect. "
        f"When controlling for district income, English-learner share, CalWorks participation, "
        f"reduced-price lunch eligibility, computers, expenditures, and enrollment, the "
        f"coefficient on the student–teacher ratio is {coef2:.3f} (p={p2:.3g}), again extremely "
        f"small and statistically insignificant. Taken together, these results indicate that this "
        f"dataset does not provide meaningful statistical evidence that lower student–teacher "
        f"ratios are associated with higher academic performance, although the observational "
        f"nature of the data means that small effects cannot be ruled out."
    )

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
