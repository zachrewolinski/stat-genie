import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity filters: keep rows with positive socket counts and non-negative AMTL
    df = df[df["sockets"] > 0].copy()
    df = df[df["num_amtl"] >= 0].copy()
    # Clip any impossible counts if present (just in case)
    df["num_amtl"] = df[["num_amtl", "sockets"]].min(axis=1)
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Indicator for modern humans vs non-human primates
    df = df.copy()
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    # Proportion of missing teeth in each tooth class/specimen
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Binomial GLM with proportion response and socket counts as frequency weights
    model = smf.glm(
        formula="prop_missing ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    return model, df


def summarize_effect(model, df: pd.DataFrame):
    # Coefficient and p-value for human effect
    coef = float(model.params.get("is_human", np.nan))
    pval = float(model.pvalues.get("is_human", np.nan))

    # Raw group-wise AMTL proportions (unadjusted)
    grouped = df.groupby("is_human").agg(
        total_amtl=("num_amtl", "sum"),
        total_sockets=("sockets", "sum"),
    )
    grouped["raw_rate"] = grouped["total_amtl"] / grouped["total_sockets"]

    raw_human = float(grouped.loc[1, "raw_rate"])
    raw_nonhuman = float(grouped.loc[0, "raw_rate"])

    # Model-based average predicted probabilities for human vs non-human
    df_pred = df.copy()
    df_pred["predicted"] = model.predict(df_pred)
    mean_pred_nonhuman = float(df_pred.loc[df_pred["is_human"] == 0, "predicted"].mean())
    mean_pred_human = float(df_pred.loc[df_pred["is_human"] == 1, "predicted"].mean())

    return {
        "coef_is_human": coef,
        "pval_is_human": pval,
        "raw_rate_human": raw_human,
        "raw_rate_nonhuman": raw_nonhuman,
        "pred_rate_human": mean_pred_human,
        "pred_rate_nonhuman": mean_pred_nonhuman,
    }


def map_to_likert(effect_summary: dict) -> int:
    coef = effect_summary["coef_is_human"]
    pval = effect_summary["pval_is_human"]
    raw_diff = effect_summary["raw_rate_human"] - effect_summary["raw_rate_nonhuman"]
    pred_diff = effect_summary["pred_rate_human"] - effect_summary["pred_rate_nonhuman"]

    # If model failed or coefficients missing, fall back to neutral
    if np.isnan(coef) or np.isnan(pval):
        return 50

    # Direction of effect is determined by both raw and adjusted differences
    direction_positive = (coef > 0) and (raw_diff > 0) and (pred_diff > 0)
    direction_negative = (coef < 0) and (raw_diff < 0) and (pred_diff < 0)

    # Magnitude of adjusted difference
    abs_pred_diff = abs(pred_diff)

    if direction_positive:
        # Strong evidence: sizable difference and highly significant
        if (abs_pred_diff >= 0.05) and (pval < 0.001):
            return 90
        if (abs_pred_diff >= 0.03) and (pval < 0.01):
            return 80
        return 70
    if direction_negative:
        if (abs_pred_diff >= 0.05) and (pval < 0.001):
            return 10
        if (abs_pred_diff >= 0.03) and (pval < 0.01):
            return 20
        return 30

    # Mixed or inconclusive evidence
    return 50


def build_explanation(effect_summary: dict, response_score: int) -> str:
    coef = effect_summary["coef_is_human"]
    pval = effect_summary["pval_is_human"]
    raw_human = effect_summary["raw_rate_human"]
    raw_nonhuman = effect_summary["raw_rate_nonhuman"]
    pred_human = effect_summary["pred_rate_human"]
    pred_nonhuman = effect_summary["pred_rate_nonhuman"]

    direction = "higher" if coef > 0 else "lower"
    strength = "strong" if response_score >= 80 or response_score <= 20 else "moderate"

    yes_no = "Yes" if response_score > 50 else "No"

    explanation = (
        f"{yes_no}.\n"
        f"I fit a binomial regression model for the proportion of missing teeth "
        f"(num_amtl / sockets) with a logit link, using socket counts as frequency weights. "
        f"The predictors were a binary indicator for modern humans vs non-human primates, "
        f"age at death, estimated probability of being male, and tooth class.\n"
        f"The coefficient for modern humans was {coef:.3f} with p-value {pval:.3g}, "
        f"indicating {strength} evidence that humans have {direction} AMTL frequencies than non-human primates "
        f"after accounting for age, sex, and tooth class.\n"
        f"Empirically, modern humans show an overall AMTL rate of approximately {raw_human:.3f}, "
        f"compared to {raw_nonhuman:.3f} for non-human primates when pooling across teeth. "
        f"Model-based average predicted AMTL probabilities are {pred_human:.3f} for humans and "
        f"{pred_nonhuman:.3f} for non-human primates, which aligns with the raw pattern.\n"
        f"The Likert-scale response of {response_score} (on a 0–100 scale, where higher values indicate a stronger "
        f"'Yes' answer) reflects both the magnitude of the human vs non-human difference and the statistical "
        f"strength of the evidence in the regression model."
    )

    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    model, df_model = fit_binomial_model(df)
    effect_summary = summarize_effect(model, df_model)
    response_score = map_to_likert(effect_summary)
    explanation = build_explanation(effect_summary, response_score)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

