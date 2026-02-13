import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Student-teacher ratio: more students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: mean of reading and math
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    return df


def simple_correlation(df: pd.DataFrame) -> float:
    return df["stratio"].corr(df["avg_score"])


def regression_effect(df: pd.DataFrame) -> tuple[float, float]:
    """
    Run a basic linear regression of avg_score on stratio and key controls.
    Returns coefficient and p-value for stratio.
    """

    # Controls for socioeconomic and demographic factors
    controls = ["income", "english", "calworks", "lunch", "expenditure", "computer", "students"]
    cols = ["avg_score", "stratio"] + controls
    df_reg = df[cols].dropna()

    y = df_reg["avg_score"]
    X = df_reg[["stratio"] + controls]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]
    return float(coef), float(pval)


def decide_answer(corr: float, coef: float, pval: float) -> tuple[str, int, str]:
    """
    Research question: "Is a lower student-teacher ratio associated with higher academic performance?"

    Lower ratio means fewer students per teacher.
    Negative correlation/coefficient between ratio and scores => Yes.
    Positive or near-zero association => No.
    """
    # Sign logic
    association_negative = (corr < 0) and (coef < 0)

    # Use p-value as evidence strength but don't hinge everything on strict 0.05
    statistically_strong = pval < 0.05

    if association_negative and statistically_strong:
        response = "Yes"
        base_conf = 80
    elif association_negative and not statistically_strong:
        response = "Yes"
        base_conf = 65
    elif not association_negative and statistically_strong:
        response = "No"
        base_conf = 80
    else:
        response = "No"
        base_conf = 60

    # Adjust confidence modestly based on magnitude of correlation
    corr_mag = abs(corr)
    if corr_mag > 0.4:
        base_conf += 10
    elif corr_mag < 0.1:
        base_conf -= 10

    confidence = int(np.clip(base_conf, 0, 100))

    explanation = (
        "I examined the relationship between student–teacher ratio (students per teacher) "
        "and average academic performance (mean of reading and math scores) across districts. "
        f"The Pearson correlation between the ratio and scores was {corr:.3f}, and the "
        f"regression coefficient on the ratio, controlling for income, poverty, English learner "
        f"share, expenditures, computers, and enrollment, was {coef:.3f} with p-value {pval:.3f}. "
        "Based on the sign and strength of these estimates, I concluded that a lower student–teacher "
        "ratio is{} associated with higher academic performance."
    )

    if response == "Yes":
        explanation = explanation.format("")
    else:
        explanation = explanation.format(" not")

    return response, confidence, explanation


def main() -> None:
    base = Path(__file__).resolve().parent
    info_path = base / "info.json"
    data_path = base / "caschools.csv"

    # Load to respect instructions, even though research question is fixed
    _meta = load_metadata(info_path)

    df = load_data(data_path)
    df = compute_str(df)

    corr = simple_correlation(df)
    coef, pval = regression_effect(df)

    response, confidence, explanation = decide_answer(corr, coef, pval)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    conclusion_path = base / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

