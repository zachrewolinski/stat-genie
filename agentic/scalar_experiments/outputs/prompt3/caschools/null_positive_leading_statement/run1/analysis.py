import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("caschools.csv")
OUTPUT_PATH = Path("conclusion.txt")


def compute_strength(corr, slope_simple, p_simple, slope_controls, p_controls, response):
    """Heuristic 0-100 strength score based on effect size and significance."""
    base = min(abs(corr) * 200.0, 100.0)

    p_min = min(p_simple, p_controls)
    if p_min < 0.001:
        sig_bonus = 15.0
    elif p_min < 0.01:
        sig_bonus = 10.0
    elif p_min < 0.05:
        sig_bonus = 5.0
    else:
        sig_bonus = -10.0

    strength = base + sig_bonus

    if (response == "Yes" and (slope_simple > 0 or slope_controls > 0 or corr > 0)) or (
        response == "No" and (slope_simple < 0 and slope_controls < 0 and corr < 0)
    ):
        strength -= 30.0

    return float(np.clip(strength, 0.0, 100.0))


def compute_confidence(df, strength):
    """Heuristic 0-100 confidence score based on sample size and strength."""
    n = len(df)
    n_factor = min((n / 500.0) * 40.0, 40.0)
    strength_factor = (strength / 100.0) * 40.0
    method_factor = 20.0
    confidence = n_factor + strength_factor + method_factor
    return float(np.clip(confidence, 0.0, 100.0))


def build_explanation(
    corr,
    slope_simple,
    p_simple,
    slope_controls,
    p_controls,
    model_simple_r2,
    model_controls_r2,
    response,
    strength,
    confidence,
):
    direction = "negative" if corr < 0 else "positive"
    explanation = (
        "Using data on 420 California K-6 and K-8 school districts, "
        "I constructed the student-teacher ratio as students divided by teachers and "
        "an overall academic performance score as the average of reading and math scores. "
        f"The correlation between student-teacher ratio and academic performance is {corr:.3f}, "
        f"indicating a {direction} association. "
        f"A simple linear regression of performance on the student-teacher ratio yields a slope of "
        f"{slope_simple:.3f} with p-value {p_simple:.3g} and R-squared {model_simple_r2:.3f}. "
        "Controlling for income, English-learner share, reduced-price lunch and CalWorks percentages, "
        "computers, and per-pupil expenditure, "
        f"the slope on the student-teacher ratio is {slope_controls:.3f} with p-value {p_controls:.3g} "
        f"and R-squared {model_controls_r2:.3f}. "
    )

    if response == "Yes":
        explanation += (
            "Both models show that higher student-teacher ratios are associated with lower test scores, "
            "and the association is statistically significant, so the data support the claim that lower "
            "student-teacher ratios are associated with higher academic performance. "
        )
    else:
        explanation += (
            "The estimated association between student-teacher ratios and test scores is weak or not "
            "statistically robust, so the data do not provide clear support for the claim that lower "
            "student-teacher ratios are associated with higher academic performance. "
        )

    explanation += (
        f"On a 0-100 scale, I rate the strength of this conclusion as {strength:.0f} and my overall "
        f"confidence, given the single observational dataset and model assumptions, as {confidence:.0f}."
    )
    return explanation


def main():
    df = pd.read_csv(DATA_PATH)

    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    corr = df["stratio"].corr(df["score"])

    x_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["score"], x_simple).fit()

    controls = ["income", "english", "lunch", "calworks", "computer", "expenditure"]
    x_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(df["score"], x_controls).fit()

    slope_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    slope_controls = float(model_controls.params["stratio"])
    p_controls = float(model_controls.pvalues["stratio"])

    if slope_simple < 0 and slope_controls < 0 and (p_simple < 0.05 or p_controls < 0.05):
        response = "Yes"
    else:
        response = "No"

    strength = compute_strength(corr, slope_simple, p_simple, slope_controls, p_controls, response)
    confidence = compute_confidence(df, strength)

    explanation = build_explanation(
        corr=corr,
        slope_simple=slope_simple,
        p_simple=p_simple,
        slope_controls=slope_controls,
        p_controls=p_controls,
        model_simple_r2=float(model_simple.rsquared),
        model_controls_r2=float(model_controls.rsquared),
        response=response,
        strength=strength,
        confidence=confidence,
    )

    result = {
        "response": response,
        "strength": int(round(strength)),
        "confidence": int(round(confidence)),
        "explanation": explanation,
    }

    OUTPUT_PATH.write_text(json.dumps(result))


if __name__ == "__main__":
    main()

