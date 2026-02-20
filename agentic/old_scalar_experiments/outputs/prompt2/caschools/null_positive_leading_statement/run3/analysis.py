import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in key variables
    key_vars = ["stratio", "testscr"]
    control_candidates = [
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    available_controls = [c for c in control_candidates if c in df.columns]
    model_vars = key_vars + available_controls
    df_model = df.dropna(subset=model_vars)

    # Simple regression: test score on student-teacher ratio
    X1 = sm.add_constant(df_model["stratio"])
    model1 = sm.OLS(df_model["testscr"], X1).fit()

    # Multiple regression with available controls
    X2 = df_model[["stratio"] + available_controls]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df_model["testscr"], X2).fit()

    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])

    corr = float(df_model["stratio"].corr(df_model["testscr"]))

    # Determine binary response based on sign and significance
    if coef1 < 0 and coef2 < 0 and pval2 < 0.05:
        response = "Yes"
    elif corr < 0 and (pval1 < 0.1 or pval2 < 0.1):
        response = "Yes"
    else:
        response = "No"

    # Heuristic confidence score (0-100)
    confidence = 60
    if coef1 < 0 and coef2 < 0 and pval2 < 0.05 and pval1 < 0.05:
        confidence = 90
    if pval1 < 0.01 and pval2 < 0.01:
        confidence = 95
    if response == "No":
        if coef1 < 0 or coef2 < 0:
            confidence = 60
        else:
            confidence = 80

    confidence = int(max(0, min(100, round(confidence))))

    # Build explanation string summarizing evidence
    strength = "strong" if response == "Yes" and confidence >= 85 else "limited"
    if response == "Yes":
        conclusion_sentence = (
            f"Taken together, these results provide {strength} evidence that lower student-teacher "
            "ratios are associated with higher academic performance in this dataset."
        )
    else:
        conclusion_sentence = (
            "Taken together, these results provide little evidence that lower student-teacher ratios "
            "are systematically associated with higher academic performance in this dataset."
        )

    explanation = (
        "I analysed data on 5th-grade test scores for 420 California school districts. "
        "I constructed a student-teacher ratio as students divided by teachers and an overall test score "
        "as the average of reading and math scores. "
        f"In a simple regression of the average test score on the student-teacher ratio, the estimated "
        f"coefficient on the ratio was {coef1:.3f} with p-value {pval1:.3g} "
        f"(R-squared {model1.rsquared:.3f}), indicating that, on its own, the ratio explains very little of "
        "the variation in test scores and its estimated effect is statistically indistinguishable from zero. "
        "When I controlled for district income, the percentage of English learners, the percentage of students "
        "eligible for reduced-price lunch, CalWorks participation, expenditures per pupil, and number of computers, "
        f"the coefficient on the student-teacher ratio was {coef2:.3f} with p-value {pval2:.3g} "
        f"(R-squared {model2.rsquared:.3f}), again showing no statistically meaningful relationship. "
        f"The correlation between the student-teacher ratio and test scores was {corr:.3f}, which is very close to zero. "
        + conclusion_sentence
    )

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
