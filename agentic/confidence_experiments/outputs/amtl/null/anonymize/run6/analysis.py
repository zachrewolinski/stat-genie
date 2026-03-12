import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing_count",
            "feature4": "socket_count",
            "feature5": "age_at_death",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    # Keep rows with valid socket counts and non-negative missing counts.
    df = df[(df["socket_count"] > 0) & (df["missing_count"] >= 0)]
    # Proportion of missing teeth in the tooth class.
    df["prop_missing"] = df["missing_count"] / df["socket_count"]
    # Indicator for modern humans vs non-human primates.
    df["is_human"] = np.where(df["genus"] == "Homo sapiens", 1.0, 0.0)
    # Center age and sex estimates for numerical stability.
    df["age_c"] = df["age_at_death"] - df["age_at_death"].mean()
    df["sex_c"] = df["sex_estimate"] - df["sex_estimate"].mean()
    return df


def fit_model(df: pd.DataFrame):
    model = smf.glm(
        formula="prop_missing ~ is_human + C(tooth_class) + age_c + sex_c",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["socket_count"],
    )
    result = model.fit()
    return result


def summarize_effect(result, df: pd.DataFrame):
    coef = result.params.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    # Predicted probability of AMTL for typical human vs non-human,
    # holding age, sex, and tooth class distribution as in the data.
    df_pred = df.copy()
    df_pred_human = df_pred.copy()
    df_pred_human["is_human"] = 1.0
    df_pred_nonhuman = df_pred.copy()
    df_pred_nonhuman["is_human"] = 0.0

    pred_human = result.predict(df_pred_human)
    pred_nonhuman = result.predict(df_pred_nonhuman)

    mean_human = float(np.mean(pred_human))
    mean_nonhuman = float(np.mean(pred_nonhuman))
    diff = mean_human - mean_nonhuman
    ratio = mean_human / mean_nonhuman if mean_nonhuman > 0 else np.nan

    return {
        "coef": float(coef),
        "pval": float(pval),
        "mean_human": mean_human,
        "mean_nonhuman": mean_nonhuman,
        "diff": diff,
        "ratio": ratio,
    }


def map_to_likert(effect_summary) -> int:
    coef = effect_summary["coef"]
    pval = effect_summary["pval"]
    ratio = effect_summary["ratio"]

    # Default to an agnostic "No" if the model fails.
    if np.isnan(coef) or np.isnan(pval) or np.isnan(ratio):
        return 50

    # If effect is not statistically significant at 0.05, treat as "No".
    if pval >= 0.05:
        # Slight adjustment based on direction of effect.
        if coef > 0:
            return 40
        elif coef < 0:
            return 20
        else:
            return 30

    # Statistically significant positive effect (humans higher AMTL).
    if coef > 0:
        if pval < 0.001 and ratio >= 1.5:
            return 90
        if pval < 0.001 and ratio >= 1.2:
            return 85
        if pval < 0.01 and ratio >= 1.2:
            return 80
        if pval < 0.05 and ratio >= 1.1:
            return 70
        # Small but significant effect.
        return 65

    # Statistically significant negative effect (humans lower AMTL).
    if coef < 0:
        if pval < 0.001 and ratio <= 0.7:
            return 10
        if pval < 0.001 and ratio <= 0.85:
            return 15
        if pval < 0.01 and ratio <= 0.9:
            return 20
        if pval < 0.05 and ratio <= 0.95:
            return 30
        return 35

    # Fallback.
    return 50


def build_explanation(effect_summary, response_score: int) -> str:
    coef = effect_summary["coef"]
    pval = effect_summary["pval"]
    mean_human = effect_summary["mean_human"]
    mean_nonhuman = effect_summary["mean_nonhuman"]
    diff = effect_summary["diff"]
    ratio = effect_summary["ratio"]

    direction = (
        "higher" if coef > 0 else "lower" if coef < 0 else "not detectably different"
    )
    significance = (
        "highly significant (p < 0.001)"
        if pval < 0.001
        else "statistically significant (p < 0.05)"
        if pval < 0.05
        else "not statistically significant (p ≥ 0.05)"
    )

    answer = "Yes" if response_score >= 50 and coef > 0 and pval < 0.05 else "No"

    explanation = (
        f"I fitted a binomial logistic regression model to the AMTL data, "
        f"using the proportion of missing teeth in each tooth class (missing_count/socket_count) "
        f"as the outcome and an indicator for modern humans (Homo sapiens vs. non-human primates), "
        f"while adjusting for estimated age at death, estimated sex, and tooth class. "
        f"The coefficient for the human indicator was {coef:.3f}, indicating that, "
        f"after accounting for age, sex, and tooth class, modern humans have {direction} "
        f"odds of antemortem tooth loss than the non-human primates in this sample. "
        f"This effect was {significance} (p = {pval:.4g}). "
        f"Based on model predictions, the mean estimated probability of a tooth being missing "
        f"for modern humans is approximately {mean_human:.3f}, compared to {mean_nonhuman:.3f} "
        f"for non-human primates, a difference of {diff:.3f} (ratio ≈ {ratio:.2f}). "
        f"Given the direction, magnitude, and statistical significance of this effect, "
        f"my overall answer to the research question "
        f"\"Do modern humans have higher frequencies of AMTL than non-human primates, "
        f"after accounting for age, sex, and tooth class?\" is \"{answer}\", "
        f"corresponding to a Likert-scale score of {response_score} on a 0–100 scale."
    )
    return explanation


def main():
    data_path = Path("amtl.csv")
    df = load_data(str(data_path))
    result = fit_model(df)
    effect_summary = summarize_effect(result, df)
    response_score = map_to_likert(effect_summary)
    explanation = build_explanation(effect_summary, response_score)

    conclusion = {"response": int(response_score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

