import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the AMTL dataset and relabel columns with their semantic meaning."""
    df = pd.read_csv(csv_path).copy()

    # According to info.json, the column names are somewhat shuffled relative to semantics.
    # We create clearer semantic aliases without dropping the originals.
    # In the supplied CSV:
    #   - 'sockets' gives tooth position (Anterior / Posterior / Premolar)
    #   - 'tooth_class' holds the taxon (Homo sapiens, Pan, Papio, Pongo)
    df["tooth_position"] = df["sockets"]
    df["specimen_id"] = df["prob_male"]  # Unique specimen identifier
    df["num_missing"] = df["genus"]  # Number of missing teeth of given class
    df["num_sockets"] = df["age"]  # Number of observable sockets
    df["age_at_death"] = df["pop"]  # Estimated age at death
    df["age_uncertainty"] = df["num_amtl"]  # Uncertainty in age estimate
    df["sex_estimate"] = df["stdev_age"]  # Sex / probability male estimate
    df["taxon"] = df["tooth_class"]  # Genus: Homo sapiens, Pan, Papio, Pongo

    # Binary human indicator for the main contrast of interest.
    df["is_human"] = (df["taxon"] == "Homo sapiens").astype(int)

    # Exclude any rows with zero sockets (would break binomial modeling).
    df = df[df["num_sockets"] > 0].copy()

    # Proportion of missing teeth per tooth-class/specimen combination.
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    """Fit a binomial GLM for AMTL with human vs. non-human indicator."""
    formula = "prop_missing ~ is_human + age_at_death + sex_estimate + C(tooth_position)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    """Summarize the human effect with p-value and predicted probabilities."""
    params = result.params
    pvalues = result.pvalues

    human_coef = params.get("is_human", np.nan)
    human_p = pvalues.get("is_human", np.nan)
    human_or = float(np.exp(human_coef)) if np.isfinite(human_coef) else np.nan

    # Predictive comparison at typical covariate levels.
    median_age = float(df["age_at_death"].median())
    median_sex = float(df["sex_estimate"].median())
    # Use the most common tooth position as reference scenario.
    common_position = df["tooth_position"].mode().iloc[0]

    new = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_at_death": [median_age, median_age],
            "sex_estimate": [median_sex, median_sex],
            "tooth_position": [common_position, common_position],
        }
    )

    pred_means = np.asarray(result.get_prediction(new).predicted_mean)
    p_nonhuman = float(pred_means[0])
    p_human = float(pred_means[1])
    diff = p_human - p_nonhuman

    return {
        "human_coef": float(human_coef),
        "human_p": float(human_p),
        "human_or": human_or,
        "p_human": p_human,
        "p_nonhuman": p_nonhuman,
        "p_diff": diff,
    }


def map_to_likert(effect_summary: dict) -> int:
    """
    Map evidence for higher human AMTL to a 0-100 Likert score.

    Heuristic:
    - If p >= 0.05, treat as insufficient evidence: scores in 0-40 range
      depending on direction and magnitude of effect.
    - If p < 0.05, treat as evidence for a relationship, scaled by p-value
      and effect size (odds ratio and predicted probability difference).
    """
    human_p = effect_summary["human_p"]
    human_or = effect_summary["human_or"]
    p_diff = effect_summary["p_diff"]

    if not np.isfinite(human_p) or not np.isfinite(human_or) or not np.isfinite(p_diff):
        return 50

    # Direction of effect: positive diff implies higher human AMTL.
    if human_p >= 0.05:
        # Weak or no evidence; Likert score biased toward "No".
        base = 20
        # If the point estimate suggests higher AMTL in humans, nudge upward a bit.
        if p_diff > 0:
            base += min(10, p_diff * 200)
        else:
            base -= min(10, abs(p_diff) * 200)
        return int(round(max(0, min(40, base))))

    # Statistically significant human effect: scale toward "Yes".
    # Convert p-value into a significance weight (lower p -> higher weight).
    sig_weight = min(1.0, max(0.0, -np.log10(human_p) / 6.0))  # cap at ~1 for tiny p-values

    # Effect-size weight from odds ratio and probability difference.
    or_weight = min(1.0, max(0.0, (human_or - 1.0) / 3.0))  # OR=4 -> weight~1
    diff_weight = min(1.0, max(0.0, abs(p_diff) / 0.25))  # 25 percentage-point diff -> weight~1

    combined = 0.4 * sig_weight + 0.3 * or_weight + 0.3 * diff_weight
    score = 50 + combined * 50  # from neutral 50 up toward 100

    # If the direction is actually opposite (humans lower AMTL), flip toward "No".
    if p_diff < 0:
        score = 50 - combined * 50

    return int(round(max(0, min(100, score))))


def build_explanation(effect_summary: dict, score: int) -> str:
    """Build a concise narrative explanation of the findings."""
    human_p = effect_summary["human_p"]
    human_or = effect_summary["human_or"]
    p_human = effect_summary["p_human"]
    p_nonhuman = effect_summary["p_nonhuman"]
    p_diff = effect_summary["p_diff"]

    direction = "higher" if p_diff > 0 else "lower"
    yes_no = "Yes" if score >= 50 else "No"

    explanation = (
        f"{yes_no}: Based on a binomial regression of the proportion of missing teeth per "
        f"specimen and tooth class (number missing over observable sockets), with predictors "
        f"for species (Homo sapiens vs. Pan/Papio/Pongo), estimated age at death, sex estimate, "
        f"and tooth class, humans show {direction} estimated AMTL frequencies than non-human primates. "
        f"The human indicator has an odds ratio of approximately {human_or:.2f} (p = {human_p:.3g}), "
        f"with predicted AMTL probabilities of about {p_nonhuman:.3%} for non-human primates and "
        f"{p_human:.3%} for humans at typical ages and across the most common tooth class, a difference "
        f"of roughly {p_diff:.3%}. This Likert score of {score} reflects both the statistical "
        f"significance (p-value) and the magnitude of the human effect on AMTL after adjusting for "
        f"age, sex, and tooth class."
    )

    return explanation


def main():
    df = load_and_prepare_data("amtl.csv")
    result = fit_binomial_model(df)
    effect_summary = summarize_effect(df, result)
    score = map_to_likert(effect_summary)
    explanation = build_explanation(effect_summary, score)

    output = {"response": int(score), "explanation": explanation}
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
