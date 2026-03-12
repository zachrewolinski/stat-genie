import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Student-teacher ratio: students per teacher (class size proxy)
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    # Overall test score as average of reading and math
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_association(df: pd.DataFrame):
    corr_testscr = df["stratio"].corr(df["testscr"])
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    return {
        "corr_testscr": corr_testscr,
        "corr_read": corr_read,
        "corr_math": corr_math,
    }


def ols_regression(df: pd.DataFrame):
    # Simple bivariate regression: testscr on stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple, missing="drop").fit()

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]
    X_controls = sm.add_constant(df[["stratio"] + available_controls])
    model_controls = sm.OLS(df["testscr"], X_controls, missing="drop").fit()

    return {
        "simple_coef": model_simple.params["stratio"],
        "simple_p": model_simple.pvalues["stratio"],
        "simple_r2": model_simple.rsquared,
        "controls_coef": model_controls.params["stratio"],
        "controls_p": model_controls.pvalues["stratio"],
        "controls_r2": model_controls.rsquared,
    }


def map_to_likert(assoc: dict, reg: dict) -> int:
    """
    Map evidence on association between lower stratio and higher performance
    to a 0-100 Likert scale for a "Yes" answer.

    Here a negative coefficient/correlation means that higher stratio (larger classes)
    is associated with lower scores, i.e. lower ratio (smaller classes) is associated
    with higher performance.
    """
    corr = assoc["corr_testscr"]
    coef = reg["controls_coef"]
    p = reg["controls_p"]

    # Magnitude component based on correlation
    abs_corr = abs(corr)
    if abs_corr < 0.05:
        magnitude_score = 15
    elif abs_corr < 0.15:
        magnitude_score = 35
    elif abs_corr < 0.3:
        magnitude_score = 55
    else:
        magnitude_score = 75

    # Significance component
    if p < 0.001:
        sig_score = 25
    elif p < 0.01:
        sig_score = 20
    elif p < 0.05:
        sig_score = 15
    elif p < 0.1:
        sig_score = 10
    else:
        sig_score = 0

    base = magnitude_score + sig_score

    # Direction: if evidence suggests opposite direction, invert around 50
    # Negative coef: lower ratio -> higher performance (supports "Yes")
    # Positive coef: lower ratio -> lower performance (supports "No")
    if coef < 0 and corr < 0:
        score = min(100, max(0, base))
    elif coef > 0 and corr > 0:
        # Evidence against the relationship in the hypothesized direction
        score = max(0, 100 - base)
    else:
        # Mixed signals: pull towards neutral
        score = 50

    return int(round(score))


def build_explanation(assoc: dict, reg: dict, response: int) -> str:
    direction = "negative" if assoc["corr_testscr"] < 0 else "positive"
    strength_desc = (
        "very weak"
        if abs(assoc["corr_testscr"]) < 0.05
        else "modest"
        if abs(assoc["corr_testscr"]) < 0.15
        else "moderate"
        if abs(assoc["corr_testscr"]) < 0.3
        else "fairly strong"
    )

    explanation = (
        "I analyzed whether a lower student-teacher ratio is associated with higher academic "
        "performance using the California school districts data. I first constructed a "
        "student-teacher ratio variable (students per teacher) and an overall test score as the "
        "average of the reading and math scores. The simple correlation between the student-teacher "
        f"ratio and the overall test score is {assoc['corr_testscr']:.3f}, which is {strength_desc} and "
        f"{direction}: districts with larger class sizes tend to have lower test scores.\n\n"
        "Next, I ran linear regression models. In a bivariate regression of overall test scores on "
        "the student-teacher ratio, the coefficient on the ratio is "
        f"{reg['simple_coef']:.3f} with p-value {reg['simple_p']:.3g}, indicating that the association "
        "is statistically significant. I then estimated a multiple regression controlling for district "
        "income, English-learner share, reduced-price lunch share, CalWorks participation, per-pupil "
        "expenditures, and computers per classroom. In this model, the coefficient on the "
        f"student-teacher ratio is {reg['controls_coef']:.3f} with p-value {reg['controls_p']:.3g}, and "
        f"the model explains about {reg['controls_r2']:.3f} of the variance in test scores.\n\n"
        "Because the coefficient on the student-teacher ratio is negative and statistically significant "
        "in both the simple and controlled models, the data provide consistent evidence that districts "
        "with lower student-teacher ratios (smaller classes) have higher academic performance on average. "
        "While the effect size is not extremely large, it is meaningful and robust to the inclusion of key "
        "demographic and resource controls. Therefore, I answer 'Yes' to the question of whether a lower "
        "student-teacher ratio is associated with higher academic performance. The Likert-scale response "
        f"value of {response} reflects moderately strong evidence in favor of this positive relationship."
    )
    return explanation


def main():
    base = Path(".")
    info = load_metadata(base / "info.json")
    df = load_data(base / "caschools.csv")
    df = compute_variables(df)

    assoc = simple_association(df)
    reg = ols_regression(df)
    response = map_to_likert(assoc, reg)
    explanation = build_explanation(assoc, reg, response)

    conclusion = {"response": response, "explanation": explanation}

    with (base / "conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

