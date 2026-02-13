import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(info_path: Path) -> dict:
    with info_path.open("r") as f:
        return json.load(f)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"students", "teachers", "read", "math"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Student-teacher ratio: higher means more students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def run_models(df: pd.DataFrame):
    # Drop rows with missing values in key variables
    cols = ["stratio", "testscr", "income", "english", "lunch", "calworks"]
    present_cols = [c for c in cols if c in df.columns]
    df_model = df[present_cols].dropna().copy()

    # Simple model: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Controlled model (if controls available)
    control_cols = [c for c in ["income", "english", "lunch", "calworks"] if c in df_model.columns]
    model_ctrl = None
    if control_cols:
        X_ctrl = sm.add_constant(df_model[["stratio"] + control_cols])
        model_ctrl = sm.OLS(y, X_ctrl).fit()

    return model_simple, model_ctrl


def summarize_results(df: pd.DataFrame, model_simple, model_ctrl):
    # Correlation between student-teacher ratio and test scores
    corr = df["stratio"].corr(df["testscr"])

    # Prefer controlled model when available
    if model_ctrl is not None:
        coef_use = model_ctrl.params["stratio"]
        p_use = float(model_ctrl.pvalues["stratio"])
        ci_low, ci_high = model_ctrl.conf_int().loc["stratio"]
        model_used = "controlled"
    else:
        coef_use = model_simple.params["stratio"]
        p_use = float(model_simple.pvalues["stratio"])
        ci_low, ci_high = model_simple.conf_int().loc["stratio"]
        model_used = "simple"

    coef_simple = model_simple.params["stratio"]
    p_simple = float(model_simple.pvalues["stratio"])

    # Decide direction of association: negative coeff means lower ratio -> higher performance
    if coef_use < 0:
        response = "Yes"
    else:
        response = "No"

    # Confidence scoring based on p-value and model consistency
    if p_use < 0.001:
        confidence = 95
    elif p_use < 0.01:
        confidence = 90
    elif p_use < 0.05:
        confidence = 80
    elif p_use < 0.1:
        confidence = 65
    else:
        confidence = 50

    # Penalize if simple and preferred model disagree on sign
    if np.sign(coef_use) != np.sign(coef_simple):
        confidence = max(0, confidence - 15)

    confidence = int(round(min(100, max(0, confidence))))

    explanation_parts = [
        "I analyzed the caschools dataset to study whether a lower student-teacher "
        "ratio is associated with higher academic performance.",
        "I constructed the student-teacher ratio as students divided by teachers "
        "and an overall test score as the average of the reading and math scores.",
        f"The Pearson correlation between the student-teacher ratio and the overall "
        f"test score was {corr:.3f}, indicating a "
        f"{'negative' if corr < 0 else 'positive' if corr > 0 else 'near-zero'} association.",
        f"In a {model_used} linear regression of test scores on the student-teacher ratio, "
        f"the estimated coefficient on the ratio was {coef_use:.3f}, with a 95% confidence "
        f"interval of [{ci_low:.3f}, {ci_high:.3f}] and a p-value of {p_use:.4f}.",
    ]

    if model_ctrl is not None:
        explanation_parts.append(
            "The controlled model included income, the percentage of English learners, "
            "the share of students on reduced-price lunch, and the CalWorks share as covariates."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because the estimated effect of the student-teacher ratio on test scores is "
            "negative in the preferred model, lower student-teacher ratios are associated "
            "with higher academic performance in this dataset."
        )
    else:
        explanation_parts.append(
            "Because the estimated effect of the student-teacher ratio on test scores is "
            "not negative in the preferred model, the data do not support the claim that "
            "lower student-teacher ratios are associated with higher academic performance."
        )

    explanation_parts.append(
        f"The confidence score of {confidence} reflects the statistical strength of the "
        "estimated association (driven mainly by the p-value) and whether simple and "
        "controlled models agree on the direction of the effect."
    )

    explanation = " ".join(explanation_parts)

    return response, confidence, explanation


def main():
    info_path = Path("info.json")
    csv_path = Path("caschools.csv")

    # Load metadata (for context, though the analysis is driven by the data file)
    if info_path.exists():
        _ = load_metadata(info_path)

    df_raw = load_data(csv_path)
    df = construct_variables(df_raw)

    model_simple, model_ctrl = run_models(df)
    response, confidence, explanation = summarize_results(df, model_simple, model_ctrl)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    # conclusion.txt must contain only this JSON object
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

