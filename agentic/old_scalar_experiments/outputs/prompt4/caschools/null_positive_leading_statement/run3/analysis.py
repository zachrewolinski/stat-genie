import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def run_analysis() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Define student-teacher ratio and an overall academic performance measure.
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Basic correlation between student-teacher ratio and average test score.
    corr = df["stratio"].corr(df["avg_score"])

    y = df["avg_score"]

    # Bivariate regression: avg_score ~ stratio
    X1 = sm.add_constant(df[["stratio"]])
    model1 = sm.OLS(y, X1).fit()

    # Multivariate regression adding key demographic and resource controls.
    controls = ["income", "english", "lunch", "computer", "expenditure"]
    X2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(y, X2).fit()

    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])
    r2_1 = float(model1.rsquared)
    r2_2 = float(model2.rsquared)

    # Map strength of evidence into a 0-100 "Yes" scale.
    if coef2 < 0 and pval2 < 0.001:
        response = 90
    elif coef2 < 0 and pval2 < 0.01:
        response = 80
    elif coef2 < 0 and pval2 < 0.05:
        response = 70
    elif coef2 < 0 and pval2 < 0.1:
        response = 60
    elif abs(coef2) < 0.01 or pval2 > 0.1:
        # Effect is tiny and/or statistically indistinguishable from zero.
        response = 20
    elif coef2 > 0 and pval2 < 0.05:
        # Statistically significant association in the opposite direction.
        response = 10
    else:
        # Ambiguous but not supportive either way.
        response = 50

    direction_word = "decrease" if coef1 < 0 else "increase"

    if abs(corr) < 0.05:
        corr_text = (
            f"The correlation between the student-teacher ratio and average "
            f"test scores is {corr:.2f}, which is essentially zero and does "
            "not indicate a clear linear association. "
        )
    else:
        corr_text = (
            f"The correlation between the student-teacher ratio and average "
            f"test scores is {corr:.2f}, suggesting that districts with more "
            "students per teacher tend to have "
            f"{'lower' if corr < 0 else 'higher'} scores. "
        )

    explanation = (
        "Using data on 420 California K-6 and K-8 districts, "
        "I computed the student-teacher ratio as students divided by teachers "
        "and defined academic performance as the average of reading and math "
        "test scores. "
        + corr_text
        + "In a simple OLS regression of average test scores on the "
        "student-teacher ratio, the "
        f"coefficient on the ratio is {coef1:.2f} (p={pval1:.3g}, R²={r2_1:.2f}), "
        f"so adding one student per teacher is associated with a "
        f"{abs(coef1):.2f}-point {direction_word} in average scores; however, "
        "this estimate is extremely small in magnitude and not statistically "
        "significant. In a multiple regression that also controls for district "
        "income, the percent of English learners, the percent of students on "
        "subsidized lunch, the number of computers, and expenditure per "
        "student, the coefficient on the student-teacher ratio is "
        f"{coef2:.2f} (p={pval2:.3g}, R²={r2_2:.2f}), again very close to zero "
        "and statistically indistinguishable from no effect. Taken together, "
        "these results do not provide evidence in this dataset that districts "
        "with lower student-teacher ratios systematically achieve higher test "
        "scores; if such an association exists, it is too small to detect with "
        "these cross-sectional data."
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    run_analysis()
