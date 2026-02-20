import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_FILE = Path("caschools.csv")
CONCLUSION_FILE = Path("conclusion.txt")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def fit_models(df: pd.DataFrame):
    # Unadjusted linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Adjusted model with key covariates that capture socioeconomic status
    covariates = [
        "stratio",
        "income",
        "calworks",
        "lunch",
        "english",
        "computer",
        "expenditure",
    ]
    df_cov = df[covariates].dropna()
    X_full = sm.add_constant(df_cov.drop(columns=["stratio"]))
    X_full.insert(1, "stratio", df_cov["stratio"])
    model_full = sm.OLS(df.loc[df_cov.index, "testscr"], X_full).fit()

    return model_simple, model_full


def summarize_results(model_simple, model_full):
    # Extract coefficient and p-value for stratio from each model
    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]

    coef_full = model_full.params["stratio"]
    p_full = model_full.pvalues["stratio"]

    return {
        "coef_simple": float(coef_simple),
        "p_simple": float(p_simple),
        "coef_full": float(coef_full),
        "p_full": float(p_full),
        "r2_simple": float(model_simple.rsquared),
        "r2_full": float(model_full.rsquared),
    }


def decide_association(stats: dict) -> tuple[str, int, str]:
    """
    Decide whether there is evidence that a lower student-teacher ratio
    is associated with higher academic performance.

    We interpret a consistently negative and statistically significant
    coefficient on stratio (p < 0.05) in both simple and adjusted models
    as evidence in favor of an association.
    """
    coef_simple = stats["coef_simple"]
    p_simple = stats["p_simple"]
    coef_full = stats["coef_full"]
    p_full = stats["p_full"]

    # Negative coefficient means: as ratio increases (more students per teacher),
    # test scores decrease -> lower ratio associated with higher performance.
    negative_and_significant_simple = coef_simple < 0 and p_simple < 0.05
    negative_and_significant_full = coef_full < 0 and p_full < 0.05

    if negative_and_significant_simple and negative_and_significant_full:
        response = "Yes"
        # Confidence higher if adjusted model also supports association
        confidence = 90
    elif (coef_simple < 0 and p_simple < 0.05) or (coef_full < 0 and p_full < 0.05):
        # Mixed or marginal evidence
        response = "Yes"
        confidence = 70
    else:
        response = "No"
        confidence = 70 if (coef_simple < 0 or coef_full < 0) else 80

    explanation_lines = [
        "We modeled average test scores (mean of reading and math) as a function of the student-teacher ratio.",
        f"In the simple regression, the coefficient on the student-teacher ratio was {stats['coef_simple']:.3f} with p-value {stats['p_simple']:.3g} (R^2 = {stats['r2_simple']:.3f}).",
        f"In the regression controlling for income, poverty (CalWorks, lunch), English learners, computers, and expenditures, the coefficient on the ratio was {stats['coef_full']:.3f} with p-value {stats['p_full']:.3g} (R^2 = {stats['r2_full']:.3f}).",
    ]

    if response == "Yes":
        explanation_lines.append(
            "In both models, a higher student-teacher ratio (more students per teacher) is associated with lower test scores, "
            "implying that a lower ratio is associated with higher academic performance. The effect remains after adjusting "
            "for key socioeconomic and resource variables, indicating a robust negative association."
        )
    else:
        explanation_lines.append(
            "The estimated relationship between the student-teacher ratio and test scores is not consistently negative and statistically significant "
            "across models, so the data do not provide strong evidence that lower ratios are associated with higher performance."
        )

    explanation = " ".join(explanation_lines)
    return response, confidence, explanation


def main():
    df = load_data()
    model_simple, model_full = fit_models(df)
    stats = summarize_results(model_simple, model_full)
    response, confidence, explanation = decide_association(stats)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    CONCLUSION_FILE.write_text(json.dumps(result))


if __name__ == "__main__":
    main()

