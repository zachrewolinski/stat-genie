import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_PATH = Path("caschools.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleaning: keep observations with positive numbers of students and teachers.
    df = df.copy()
    df = df[(df["students"] > 0) & (df["teachers"] > 0)]

    # Construct student–teacher ratio and overall test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in the key variables.
    df = df.dropna(subset=["stratio", "testscr"])
    return df


def analyze_relationship(df: pd.DataFrame):
    # Correlation between student–teacher ratio and test scores.
    corr = df["stratio"].corr(df["testscr"])

    # Simple linear regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df).fit()

    # Multiple regression controlling for key demographics / resources where available.
    controls = []
    for col in ["income", "calworks", "lunch", "english", "expenditure"]:
        if col in df.columns:
            controls.append(col)

    if controls:
        formula = "testscr ~ stratio + " + " + ".join(controls)
        model_controls = smf.ols(formula, data=df).fit()
    else:
        model_controls = None

    results = {
        "corr": float(corr),
        "corr_n": int(df.shape[0]),
        "simple_coef": float(model_simple.params.get("stratio", np.nan)),
        "simple_pvalue": float(model_simple.pvalues.get("stratio", np.nan)),
        "simple_r2": float(model_simple.rsquared),
        "controls": controls,
    }

    if model_controls is not None:
        results.update(
            {
                "controls_coef": float(model_controls.params.get("stratio", np.nan)),
                "controls_pvalue": float(
                    model_controls.pvalues.get("stratio", np.nan)
                ),
                "controls_r2": float(model_controls.rsquared),
            }
        )

    return results


def derive_conclusion(stats: dict):
    corr = stats["corr"]
    simple_coef = stats["simple_coef"]
    simple_p = stats["simple_pvalue"]
    controls_coef = stats.get("controls_coef")
    controls_p = stats.get("controls_pvalue")
    n = stats["corr_n"]

    # Determine direction and robustness of the association.
    # Lower student–teacher ratio should correspond to higher scores,
    # so we expect a NEGATIVE relationship between ratio and test scores.
    negative_direction = (corr < 0) and (simple_coef < 0)

    significant_simple = simple_p is not None and simple_p < 0.05
    significant_controls = (
        controls_p is not None and not np.isnan(controls_p) and controls_p < 0.05
    )

    if negative_direction and (significant_simple or significant_controls):
        response = "Yes"
    else:
        response = "No"

    # Strength reflects both magnitude and robustness of the evidence.
    abs_corr = abs(corr)
    base_strength = min(100, int(abs_corr * 100))

    if response == "Yes":
        if significant_simple and significant_controls:
            strength = max(base_strength, 70)
        elif significant_simple or significant_controls:
            strength = max(base_strength, 55)
        else:
            strength = max(base_strength, 35)
    else:
        # Either weak or no evidence for the expected negative association.
        if abs_corr < 0.05 and not significant_simple and not significant_controls:
            strength = 70
        else:
            strength = 50

    # Confidence reflects data quality and model consistency.
    if n >= 400 and abs_corr >= 0.2 and (significant_simple or significant_controls):
        confidence = 85
    elif n >= 300 and (significant_simple or significant_controls):
        confidence = 75
    else:
        confidence = 60

    # Build a concise explanation summarizing the evidence.
    explanation_parts = [
        f"The analysis uses {n} school districts from the caschools dataset.",
        f"The student–teacher ratio has a correlation of {corr:.3f} with average test scores (testscr = (read + math)/2).",
        f"In a simple linear regression of testscr on the student–teacher ratio, the coefficient on the ratio is {simple_coef:.3f} with p-value {simple_p:.3g}.",
    ]

    controls = stats.get("controls", [])
    if controls and stats.get("controls_coef") is not None:
        explanation_parts.append(
            f"Controlling for {', '.join(controls)}, the coefficient on the student–teacher ratio is {controls_coef:.3f} with p-value {controls_p:.3g}."
        )

    if response == "Yes":
        explanation_parts.append(
            "Both the sign and statistical significance of the association indicate that lower student–teacher ratios are associated with higher academic performance."
        )
    else:
        explanation_parts.append(
            "The estimated relationship between student–teacher ratio and test scores is not consistently negative and statistically significant, so the data do not provide strong evidence that lower ratios are associated with higher academic performance."
        )

    explanation_parts.append(
        f"The strength score ({strength}) reflects the magnitude and robustness of the estimated association, "
        f"and the confidence score ({confidence}) reflects the sample size and consistency of results across models."
    )

    explanation = " ".join(explanation_parts)

    return response, strength, confidence, explanation


def main():
    df = load_and_prepare_data(DATA_PATH)
    stats = analyze_relationship(df)
    response, strength, confidence, explanation = derive_conclusion(stats)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    CONCLUSION_PATH.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

