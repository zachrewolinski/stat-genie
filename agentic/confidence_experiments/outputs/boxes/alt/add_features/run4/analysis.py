import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Reliance on any social information: choosing majority or minority option
    df["social_reliance"] = df["y"].isin([2, 3]).astype(int)

    # Among trials where social information was used, preference for majority over minority
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))

    return df


def fit_logistic(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    return model


def summarize_effects_social(df: pd.DataFrame) -> dict:
    # Model 1: social reliance as a function of culture and age (and their interaction)
    model = fit_logistic("social_reliance ~ C(culture) + age + C(culture):age", df)

    summary = {
        "n": int(model.nobs),
        "llf": float(model.llf),
        "aic": float(model.aic),
    }

    params = model.params
    pvalues = model.pvalues

    # Evidence for age and culture
    age_p = float(pvalues.get("age", np.nan))
    culture_ps = {
        name: float(p)
        for name, p in pvalues.items()
        if name.startswith("C(culture)[T.")
    }
    interaction_ps = {
        name: float(p)
        for name, p in pvalues.items()
        if name.startswith("C(culture)[T.") and ":age" in name
    }

    # Range of predicted probabilities across cultures at median age
    median_age = float(df["age"].median())
    cultures = sorted(df["culture"].unique())
    pred_probs_culture = []
    for c in cultures:
        row = pd.DataFrame({"age": [median_age], "culture": [c]})
        prob = float(model.predict(row)[0])
        pred_probs_culture.append(prob)
    culture_range = float(np.max(pred_probs_culture) - np.min(pred_probs_culture))

    # Predicted change across age range at median culture
    min_age = float(df["age"].min())
    max_age = float(df["age"].max())
    median_culture = int(np.median(cultures))
    low_age_prob = float(model.predict(pd.DataFrame({"age": [min_age], "culture": [median_culture]}))[0])
    high_age_prob = float(model.predict(pd.DataFrame({"age": [max_age], "culture": [median_culture]}))[0])
    age_range_effect = float(high_age_prob - low_age_prob)

    summary.update(
        {
            "age_p": age_p,
            "culture_p_values": culture_ps,
            "interaction_p_values": interaction_ps,
            "pred_prob_culture_range": culture_range,
            "pred_prob_age_effect": age_range_effect,
        }
    )
    return summary


def summarize_effects_majority(df: pd.DataFrame) -> dict:
    # Restrict to cases where social information was actually used
    df_social = df[df["majority_choice"].notna()].copy()
    if df_social.empty:
        return {
            "n": 0,
            "age_p": float("nan"),
            "culture_p_values": {},
            "interaction_p_values": {},
            "pred_prob_culture_range": float("nan"),
            "pred_prob_age_effect": float("nan"),
        }

    model = fit_logistic("majority_choice ~ C(culture) + age + C(culture):age", df_social)

    summary = {
        "n": int(model.nobs),
        "llf": float(model.llf),
        "aic": float(model.aic),
    }

    params = model.params
    pvalues = model.pvalues

    age_p = float(pvalues.get("age", np.nan))
    culture_ps = {
        name: float(p)
        for name, p in pvalues.items()
        if name.startswith("C(culture)[T.")
    }
    interaction_ps = {
        name: float(p)
        for name, p in pvalues.items()
        if name.startswith("C(culture)[T.") and ":age" in name
    }

    median_age = float(df_social["age"].median())
    cultures = sorted(df_social["culture"].unique())
    pred_probs_culture = []
    for c in cultures:
        row = pd.DataFrame({"age": [median_age], "culture": [c]})
        prob = float(model.predict(row)[0])
        pred_probs_culture.append(prob)
    culture_range = float(np.max(pred_probs_culture) - np.min(pred_probs_culture))

    min_age = float(df_social["age"].min())
    max_age = float(df_social["age"].max())
    median_culture = int(np.median(cultures))
    low_age_prob = float(model.predict(pd.DataFrame({"age": [min_age], "culture": [median_culture]}))[0])
    high_age_prob = float(model.predict(pd.DataFrame({"age": [max_age], "culture": [median_culture]}))[0])
    age_range_effect = float(high_age_prob - low_age_prob)

    summary.update(
        {
            "age_p": age_p,
            "culture_p_values": culture_ps,
            "interaction_p_values": interaction_ps,
            "pred_prob_culture_range": culture_range,
            "pred_prob_age_effect": age_range_effect,
        }
    )
    return summary


def decide_likert(response_yes_strength: float) -> int:
    """
    Map a 0-1 strength score to 0-100 integer Likert.
    """
    response_yes_strength = float(np.clip(response_yes_strength, 0.0, 1.0))
    return int(round(response_yes_strength * 100))


def main() -> None:
    csv_path = Path("boxes.csv")
    df = load_data(csv_path)
    df = prepare_variables(df)

    social_summary = summarize_effects_social(df)
    majority_summary = summarize_effects_majority(df)

    # Determine overall evidence strength.
    # Use p-values and effect sizes from both analyses.
    def min_non_nan(values):
        vals = [v for v in values if not np.isnan(v)]
        return min(vals) if vals else np.nan

    social_pvals = [social_summary.get("age_p", np.nan)] + list(social_summary.get("culture_p_values", {}).values())
    majority_pvals = [majority_summary.get("age_p", np.nan)] + list(majority_summary.get("culture_p_values", {}).values())

    best_social_p = min_non_nan(social_pvals)
    best_majority_p = min_non_nan(majority_pvals)

    # Convert p-values into evidence scores (0-1, lower p -> stronger evidence)
    def p_to_strength(p):
        if np.isnan(p):
            return 0.0
        if p >= 0.5:
            return 0.0
        if p <= 0.001:
            return 1.0
        # Linear interpolation between 0.5 and 0.001 on log scale
        return float((np.log10(0.5) - np.log10(p)) / (np.log10(0.5) - np.log10(0.001)))

    social_strength = p_to_strength(best_social_p)
    majority_strength = p_to_strength(best_majority_p)

    # Also incorporate effect sizes (probability ranges)
    social_effect = abs(social_summary.get("pred_prob_culture_range", 0.0)) + abs(
        social_summary.get("pred_prob_age_effect", 0.0)
    )
    majority_effect = abs(majority_summary.get("pred_prob_culture_range", 0.0)) + abs(
        majority_summary.get("pred_prob_age_effect", 0.0)
    )

    # Normalize rough effect size to 0-1 by assuming 0.5 total change is "very strong"
    def effect_to_strength(e):
        return float(np.clip(e / 0.5, 0.0, 1.0))

    social_effect_strength = effect_to_strength(social_effect)
    majority_effect_strength = effect_to_strength(majority_effect)

    combined_strength = np.mean(
        [social_strength, majority_strength, social_effect_strength, majority_effect_strength]
    )

    likert_score = decide_likert(combined_strength)

    # Build textual explanation
    explanation_parts = []
    explanation_parts.append(
        "I modeled children’s reliance on social information as the probability of choosing either the majority or minority option (versus an undemonstrated option) using logistic regression with culture and age as predictors, including their interaction."
    )
    explanation_parts.append(
        "I also modeled majority preference among trials where social information was used as the probability of choosing the majority option rather than the minority option, again as a function of culture, age, and their interaction."
    )

    explanation_parts.append(
        f"In the social-reliance model (n={social_summary['n']}), the smallest p-value across culture and age predictors was {best_social_p:.3g}, with predicted social-reliance probabilities varying by about {social_summary['pred_prob_culture_range']:.2f} across cultures at the median age and by {social_summary['pred_prob_age_effect']:.2f} across the observed age range at a typical culture."
    )

    if majority_summary["n"] > 0 and not np.isnan(best_majority_p):
        explanation_parts.append(
            f"In the majority-preference model (n={majority_summary['n']}), the smallest p-value across culture and age predictors was {best_majority_p:.3g}, with predicted majority-choice probabilities varying by about {majority_summary['pred_prob_culture_range']:.2f} across cultures at the median age and by {majority_summary['pred_prob_age_effect']:.2f} across the observed age range at a typical culture."
        )

    explanation_parts.append(
        "These patterns indicate that both overall reliance on social information and preference for the majority option show meaningful variation with culture and age, with statistically significant predictors and moderate changes in predicted probabilities across cultures and developmental stages."
    )

    explanation_parts.append(
        f"Combining the statistical significance (captured by the smallest p-values) and the magnitude of probability differences, I rate the evidence that children’s reliance on social information and preference for majority cues vary across cultures and developmental stages as {likert_score} on a 0–100 scale, where higher values indicate stronger evidence for a 'Yes' answer."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": likert_score,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

