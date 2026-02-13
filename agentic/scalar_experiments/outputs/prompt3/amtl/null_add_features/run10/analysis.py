import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only columns relevant to the AMTL analysis
    keep_cols = [
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "genus",
        "pop",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].copy()

    # Basic sanity filters: drop rows with missing key fields or non-positive sockets
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[df["sockets"] > 0].copy()

    # Create indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure tooth_class is treated as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression of AMTL counts on genus (human vs non-human),
    controlling for age, sex (prob_male), and tooth class.
    """
    # Response as proportion with binomial weights
    y = df["num_amtl"] / df["sockets"]
    weights = df["sockets"]

    # Design matrix: intercept, is_human, age, prob_male, tooth class dummies
    X = df[["is_human", "age", "prob_male", "tooth_class"]].copy()
    X = pd.get_dummies(X, columns=["tooth_class"], drop_first=True)
    X = sm.add_constant(X, prepend=True)

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    return result, X.columns


def summarize_effect(result) -> dict:
    params = result.params
    bse = result.bse

    # Extract human effect
    if "is_human" not in params.index:
        raise ValueError("Model is missing 'is_human' coefficient.")

    beta = params["is_human"]
    se = bse["is_human"]

    # 95% Wald confidence interval on log-odds scale
    z = 1.96
    ci_low = beta - z * se
    ci_high = beta + z * se

    # Convert to odds ratio for interpretability
    odds_ratio = float(np.exp(beta))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    # Two-sided p-value for null of no human effect
    p_value = float(result.pvalues["is_human"])

    return {
        "beta": float(beta),
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "odds_ratio": odds_ratio,
        "or_low": or_low,
        "or_high": or_high,
        "p_value": p_value,
    }


def decide_answer(effect: dict) -> dict:
    """
    Map the estimated human effect into a Yes/No answer,
    a strength score (0-100), and a confidence score (0-100).
    """
    beta = effect["beta"]
    ci_low = effect["ci_low"]
    ci_high = effect["ci_high"]
    p_value = effect["p_value"]
    odds_ratio = effect["odds_ratio"]

    # Directional conclusion based on point estimate and CI
    if ci_low > 0:
        response = "Yes"
    elif ci_high < 0:
        response = "No"
    else:
        # Inconclusive interval: fall back to sign of beta
        response = "Yes" if beta > 0 else "No"

    # Strength reflects both effect size and statistical certainty
    # Normalize |log(OR)| via a simple saturating transform
    effect_magnitude = abs(np.log(odds_ratio))
    effect_score = 100 * (1 - np.exp(-effect_magnitude))

    # Confidence primarily driven by CI excluding 0 and by p-value
    if (ci_low > 0) or (ci_high < 0):
        # Clear separation from 0
        conf_from_p = max(0.0, 1.0 - p_value)
        confidence = 70 + 30 * conf_from_p
    else:
        # CI overlaps 0: more modest confidence
        conf_from_p = max(0.0, 1.0 - min(p_value, 0.5) / 0.5)
        confidence = 40 + 40 * conf_from_p

    confidence = float(max(0.0, min(100.0, confidence)))
    strength = float(max(0.0, min(100.0, effect_score)))

    return {
        "response": response,
        "strength": round(strength, 1),
        "confidence": round(confidence, 1),
    }


def build_explanation(effect: dict, decision: dict) -> str:
    direction = "higher" if decision["response"] == "Yes" else "not higher"
    explanation = (
        "I fit a binomial regression model predicting the proportion of missing teeth "
        "(num_amtl out of sockets) from an indicator for modern humans versus non-human primates, "
        "while adjusting for age, sex (probability of being male), and tooth class (anterior, posterior, premolar). "
        "The coefficient for the human indicator on the log-odds scale was "
        f"{effect['beta']:.3f} with a 95% Wald confidence interval from "
        f"{effect['ci_low']:.3f} to {effect['ci_high']:.3f}, corresponding to an odds ratio of "
        f"{effect['odds_ratio']:.2f} (95% CI {effect['or_low']:.2f}–{effect['or_high']:.2f}) and a p-value of "
        f"{effect['p_value']:.3g}. "
        f"This indicates that, after accounting for age, sex, and tooth class, modern humans have {direction} "
        "frequencies of antemortem tooth loss compared to the combined non-human primate genera (Pan, Pongo, Papio). "
        "The reported strength reflects the magnitude of the estimated human effect on the odds of tooth loss, "
        "and the confidence score reflects how clearly the confidence interval and p-value distinguish the human effect "
        "from zero."
    )
    return explanation


def main():
    data_path = Path("amtl.csv")
    df = load_data(data_path)

    result, _ = fit_binomial_model(df)
    effect = summarize_effect(result)
    decision = decide_answer(effect)
    explanation = build_explanation(effect, decision)

    conclusion = {
        "response": decision["response"],
        "strength": decision["strength"],
        "confidence": decision["confidence"],
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

