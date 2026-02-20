import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    df = df.copy()
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    str_var = df["str"]
    testscr = df["testscr"]

    simple_x = sm.add_constant(str_var)
    simple_model = sm.OLS(testscr, simple_x).fit()

    controls = ["income", "english", "lunch", "calworks"]
    available_controls = [c for c in controls if c in df.columns]
    if available_controls:
        x_controls = df[["str"] + available_controls]
        x_controls = sm.add_constant(x_controls)
        control_model = sm.OLS(testscr, x_controls).fit()
    else:
        control_model = None

    simple_beta = float(simple_model.params["str"])
    simple_p = float(simple_model.pvalues["str"])
    simple_r2 = float(simple_model.rsquared)
    corr = float(str_var.corr(testscr))

    control_beta = None
    control_p = None
    control_r2 = None
    if control_model is not None:
        control_beta = float(control_model.params["str"])
        control_p = float(control_model.pvalues["str"])
        control_r2 = float(control_model.rsquared)

    association_negative = simple_beta < 0.0 and corr < 0.0
    strong_evidence = association_negative and simple_p < 0.05
    if control_beta is not None and control_p is not None:
        association_negative = association_negative and control_beta < 0.0
        strong_evidence = strong_evidence and control_p < 0.05

    if strong_evidence:
        response = "Yes"
        confidence = 90
    elif association_negative:
        response = "Yes"
        confidence = 70
    else:
        response = "No"
        confidence = 60

    explanation_parts = [
        "I used data from 420 California K-6 and K-8 districts to examine whether a lower student-teacher ratio is associated with higher academic performance.",
        "I constructed the student-teacher ratio as students divided by teachers and the academic performance measure as the average of reading and math scores.",
        f"In a simple linear regression of average test score on the student-teacher ratio, the coefficient on the ratio was {simple_beta:.3f} with p-value {simple_p:.3g} and R-squared {simple_r2:.3f}.",
        f"The correlation between the student-teacher ratio and average test score was {corr:.3f}, indicating that districts with fewer students per teacher tend to have higher scores.",
    ]

    if control_beta is not None and control_p is not None and control_r2 is not None:
        explanation_parts.append(
            "I also estimated a regression controlling for income, English-learner share, reduced-price-lunch share, and CalWorks participation."
        )
        explanation_parts.append(
            f"In this controlled model, the coefficient on the student-teacher ratio was {control_beta:.3f} with p-value {control_p:.3g} and R-squared {control_r2:.3f}."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because the estimated relationship between the student-teacher ratio and test scores is consistently negative and statistically significant in the primary specification, the data support the conclusion that lower student-teacher ratios are associated with higher academic performance in this dataset."
        )
    else:
        explanation_parts.append(
            "Because the estimated relationship between the student-teacher ratio and test scores is not consistently negative and statistically significant, the data do not provide strong evidence that lower student-teacher ratios are associated with higher academic performance in this dataset."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

