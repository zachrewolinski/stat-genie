import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Construct student–teacher ratio (students per teacher) and overall test score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing or non-finite values in key variables.
    key_cols = ["stratio", "testscr", "income", "lunch", "calworks", "english", "expenditure"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=key_cols)
    return df


def analyze_relationship(df: pd.DataFrame):
    # Simple Pearson correlation between student–teacher ratio and test scores.
    r_simple, p_simple = pearsonr(df["stratio"], df["testscr"])

    # Simple OLS: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multiple OLS controlling for key demographics and resources.
    controls = ["income", "lunch", "calworks", "english", "expenditure"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()
    coef_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]

    results = {
        "r_simple": float(r_simple),
        "p_simple": float(p_simple),
        "coef_simple": float(coef_simple),
        "pval_simple": float(pval_simple),
        "coef_multi": float(coef_multi),
        "pval_multi": float(pval_multi),
        "r2_simple": float(model_simple.rsquared),
        "r2_multi": float(model_multi.rsquared),
    }
    return results


def decide_answer(stats: dict):
    coef_multi = stats["coef_multi"]
    pval_multi = stats["pval_multi"]
    r_simple = stats["r_simple"]

    # By construction, lower student–teacher ratio corresponds to smaller stratio.
    # A negative association (higher scores as stratio falls) means coef_multi < 0.
    if coef_multi < 0 and pval_multi < 0.05:
        response = "Yes"
        direction_text = "a lower student–teacher ratio is associated with higher test scores"
    else:
        response = "No"
        if coef_multi >= 0:
            direction_text = "there is no evidence that smaller classes are associated with higher scores; the estimated association is non-negative"
        else:
            direction_text = "the estimated association is weak or statistically indistinguishable from zero once controls are included"

    # Map strength and confidence heuristically from p-value and correlation magnitude.
    # Focus on the controlled regression (pval_multi) but incorporate the simple correlation.
    abs_r = abs(r_simple)
    if pval_multi < 1e-6:
        base_strength = 90
    elif pval_multi < 1e-3:
        base_strength = 80
    elif pval_multi < 0.01:
        base_strength = 70
    elif pval_multi < 0.05:
        base_strength = 60
    elif pval_multi < 0.1:
        base_strength = 50
    else:
        base_strength = 40

    # Adjust slightly based on effect size.
    if abs_r > 0.4:
        base_strength += 5
    elif abs_r < 0.1:
        base_strength -= 5

    strength = int(max(0, min(100, base_strength)))
    # Confidence is similar but slightly more conservative.
    confidence = int(max(0, min(100, base_strength - 5)))

    return response, strength, confidence, direction_text


def build_explanation(stats: dict, direction_text: str) -> str:
    explanation = (
        "Using data on 420 California school districts, I constructed the student–teacher ratio "
        "as students per teacher and an overall achievement measure as the average of reading and math scores. "
        f"The simple Pearson correlation between the student–teacher ratio and average test score is {stats['r_simple']:.3f} "
        f"(p = {stats['p_simple']:.3g}), and a simple OLS regression of test scores on the ratio yields a coefficient of "
        f"{stats['coef_simple']:.3f} (p = {stats['pval_simple']:.3g}, R² = {stats['r2_simple']:.3f}). "
        "To account for observable differences across districts, I then estimated a multiple regression of test scores on "
        "the student–teacher ratio controlling for average district income, the shares of students on CalWorks, reduced-price lunch, "
        "and English learners, and per-pupil expenditures. "
        f"In this controlled model, the coefficient on the student–teacher ratio is {stats['coef_multi']:.3f} "
        f"(p = {stats['pval_multi']:.3g}, R² = {stats['r2_multi']:.3f}). "
        f"Because of these results, I conclude that {direction_text}. "
        "This conclusion is based on associations in observational data; while the relationship is statistically characterized, "
        "it should not be interpreted as a fully causal effect without stronger identification."
    )
    return explanation


def main():
    csv_path = Path("caschools.csv")
    df = load_data(csv_path)
    stats = analyze_relationship(df)
    response, strength, confidence, direction_text = decide_answer(stats)
    explanation = build_explanation(stats, direction_text)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

