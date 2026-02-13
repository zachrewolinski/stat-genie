import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop any rows with clearly invalid socket counts
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression for AMTL proportion with:
    - genus (Homo sapiens vs Pan/Pongo/Papio)
    - age
    - prob_male (sex proxy)
    - tooth_class
    using a logit link and sockets as trial counts.
    """

    df = df.copy()
    # Proportion of missing teeth
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Set clear reference categories:
    # Use non-human primates as baseline by excluding Homo from reference
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Categorical predictors
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Use GLM with binomial family and weights equal to number of sockets
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    return model


def summarize_effect(model) -> dict:
    """
    Extract the human effect estimate and its uncertainty on the log-odds scale
    and convert to an interpretable effect size.
    """
    params = model.params
    bse = model.bse

    if "is_human" not in params:
        raise RuntimeError("Model does not contain is_human coefficient.")

    coef = params["is_human"]
    se = bse["is_human"]

    # 95% Wald CI on log-odds
    z = 1.96
    lower = coef - z * se
    upper = coef + z * se

    # Convert to odds ratios
    or_est = float(np.exp(coef))
    or_low = float(np.exp(lower))
    or_high = float(np.exp(upper))

    # Approximate p-value from z statistic using normal approximation
    z_stat = coef / se if se > 0 else np.nan
    p_value = float(2 * (1 - norm.cdf(abs(z_stat)))) if se > 0 and np.isfinite(z_stat) else float("nan")

    return {
        "coef_log_odds": float(coef),
        "se": float(se),
        "ci95_log_odds": [float(lower), float(upper)],
        "odds_ratio": or_est,
        "ci95_odds_ratio": [or_low, or_high],
        "z_stat": float(z_stat),
        "p_value": p_value,
    }


def map_effect_to_likert(effect_summary: dict) -> int:
    """
    Map the evidence about the human effect to a 0–100 scalar.

    Heuristic:
    - If odds_ratio ~ 1 and CI includes 1 with weak evidence, map near 50.
    - Strong positive effect (OR >> 1, CI above 1, small p) -> closer to 100.
    - Strong negative effect (OR << 1, CI below 1, small p) -> closer to 0.
    """
    or_est = effect_summary["odds_ratio"]
    or_low, or_high = effect_summary["ci95_odds_ratio"]
    p = effect_summary["p_value"]

    # Evidence strength component based on p-value
    if np.isnan(p):
        evidence_strength = 0.0
    elif p < 0.001:
        evidence_strength = 1.0
    elif p < 0.01:
        evidence_strength = 0.8
    elif p < 0.05:
        evidence_strength = 0.6
    elif p < 0.1:
        evidence_strength = 0.4
    else:
        evidence_strength = 0.2

    # Direction component based on whether the CI is mostly above or below 1
    if or_low > 1.0:
        direction = 1.0  # clearly positive
    elif or_high < 1.0:
        direction = -1.0  # clearly negative
    else:
        # Mixed/overlapping CI: direction proportional to log(or_est)
        direction = float(np.tanh(np.log(or_est)))

    # Map direction [-1,1] and evidence_strength [0,1] into [0,100]
    # Center is 50; positive pushes up, negative down.
    score = 50 + 50 * direction * evidence_strength
    # Ensure valid integer in [0,100]
    score_int = int(round(min(100, max(0, score))))
    return score_int


def build_explanation(effect_summary: dict, score: int) -> str:
    or_est = effect_summary["odds_ratio"]
    or_low, or_high = effect_summary["ci95_odds_ratio"]
    p = effect_summary["p_value"]

    if score > 55:
        qualitative = "Yes, the analysis supports higher AMTL frequencies in modern humans relative to non-human primates after adjusting for age, sex, and tooth class."
    elif score < 45:
        qualitative = "No, the analysis does not support higher AMTL frequencies in modern humans relative to non-human primates after adjusting for age, sex, and tooth class."
    else:
        qualitative = "The analysis provides inconclusive evidence that modern humans have higher AMTL frequencies than non-human primates after adjusting for age, sex, and tooth class."

    explanation = (
        f"{qualitative} "
        f"In a binomial regression of the proportion of missing teeth (num_amtl / sockets) on genus, age, sex (prob_male), "
        f"and tooth_class, the coefficient for humans (is_human) corresponds to an odds ratio of approximately "
        f"{or_est:.2f} (95% CI {or_low:.2f}–{or_high:.2f}, p ≈ {p:.3g}). "
        f"This means that, holding age, sex, and tooth class constant, humans have roughly {or_est:.2f}-fold odds of antemortem "
        f"tooth loss compared to non-human primates. The Likert-scale response score of {score} reflects both the direction and "
        f"statistical strength of this estimated effect."
    )
    return explanation


def main():
    data_path = Path("amtl.csv")
    df = load_data(str(data_path))
    model = fit_binomial_model(df)
    effect_summary = summarize_effect(model)
    score = map_effect_to_likert(effect_summary)
    explanation = build_explanation(effect_summary, score)

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
