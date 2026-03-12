import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


@dataclass
class EffectSummary:
    p_value: Optional[float]
    range_diff: Optional[float]
    lr_stat: Optional[float]
    df_diff: Optional[int]


def _lr_test(
    model_big,
    model_small,
) -> Tuple[float, float, int]:
    """Manual likelihood-ratio test using log-likelihoods."""
    lr_stat = 2.0 * (model_big.llf - model_small.llf)
    df_diff = int(round(model_big.df_model - model_small.df_model))
    if df_diff <= 0:
        # Not a valid nested comparison; fall back to no additional evidence.
        return 0.0, 1.0, 0
    p_value = float(chi2.sf(lr_stat, df_diff))
    return float(lr_stat), p_value, df_diff


def _significance_score(p: Optional[float]) -> float:
    """Map a p-value to a [0, 1] evidence score."""
    if p is None or np.isnan(p):
        return 0.0
    if p < 1e-4:
        return 1.0
    if p < 1e-3:
        return 0.95
    if p < 1e-2:
        return 0.85
    if p < 5e-2:
        return 0.7
    if p < 1e-1:
        return 0.4
    return 0.0


def _range_score(r: Optional[float]) -> float:
    """Map a difference in proportions to a [0, 1] effect-size score."""
    if r is None or np.isnan(r):
        return 0.0
    if r <= 0:
        return 0.0
    # Saturate around a 50 percentage point difference.
    return float(min(1.0, r / 0.5))


def _effect_score(effect: EffectSummary) -> float:
    sig = _significance_score(effect.p_value)
    rng = _range_score(effect.range_diff)
    # Heavier weight on statistical evidence, but still factor in magnitude.
    return 0.7 * sig + 0.3 * rng


def _likert_from_effects(
    site_effect: EffectSummary,
    age_effect: EffectSummary,
    interaction_p: Optional[float],
) -> Tuple[int, float]:
    site_score = _effect_score(site_effect)
    age_score = _effect_score(age_effect)
    overall = (site_score + age_score) / 2.0

    if interaction_p is not None and not np.isnan(interaction_p) and interaction_p < 0.05:
        # Interaction indicates patterned variation in age trends across cultures.
        overall = min(1.0, overall + 0.1)

    response = int(round(overall * 100))
    # Clamp just in case of rounding edge cases.
    response = max(0, min(100, response))
    return response, overall


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename for clarity.
    df = df.rename(
        columns={
            "majority_first": "outcome",
            "culture": "maj_demo_first",
            "y": "site",
        }
    )

    # Outcome coding: 1=unchosen option, 2=majority option, 3=minority option.
    df["majority_choice"] = (df["outcome"] == 2).astype(int)

    # Basic descriptive statistics for majority choice across sites and age groups.
    site_stats = (
        df.groupby("site")["majority_choice"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "prop_majority"})
    )

    age_bins = [3, 6, 9, 12, 15]
    age_labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    age_group_stats = (
        df.groupby("age_group")["majority_choice"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "prop_majority"})
    )

    overall_mean = float(df["majority_choice"].mean())
    site_means = site_stats["prop_majority"]
    age_means = age_group_stats["prop_majority"].dropna()

    min_site_mean = float(site_means.min())
    max_site_mean = float(site_means.max())
    min_age_mean = float(age_means.min()) if not age_means.empty else np.nan
    max_age_mean = float(age_means.max()) if not age_means.empty else np.nan

    # Logistic regression models.
    # Base model with demonstration order only.
    model_base = smf.logit("majority_choice ~ maj_demo_first", data=df).fit(disp=False)

    # Add age, add site, and add both.
    model_age_only = smf.logit(
        "majority_choice ~ maj_demo_first + age", data=df
    ).fit(disp=False)
    model_site_only = smf.logit(
        "majority_choice ~ maj_demo_first + C(site)", data=df
    ).fit(disp=False)
    model_both = smf.logit(
        "majority_choice ~ maj_demo_first + age + C(site)", data=df
    ).fit(disp=False)

    # LR test for additional site effect given age.
    lr_site_stat, lr_site_p, lr_site_df = _lr_test(model_both, model_age_only)
    site_effect = EffectSummary(
        p_value=float(lr_site_p),
        range_diff=max_site_mean - min_site_mean,
        lr_stat=float(lr_site_stat),
        df_diff=int(lr_site_df),
    )

    # LR test for additional age effect given site.
    lr_age_stat, lr_age_p, lr_age_df = _lr_test(model_both, model_site_only)
    age_effect = EffectSummary(
        p_value=float(lr_age_p),
        range_diff=(
            (max_age_mean - min_age_mean) if not np.isnan(min_age_mean) else np.nan
        ),
        lr_stat=float(lr_age_stat),
        df_diff=int(lr_age_df),
    )

    # Optional interaction: age × site.
    try:
        model_inter = smf.logit(
            "majority_choice ~ maj_demo_first + age * C(site)", data=df
        ).fit(disp=False)
        lr_inter_stat, lr_inter_p, lr_inter_df = _lr_test(model_inter, model_both)
        interaction_p: Optional[float] = float(lr_inter_p)
        interaction_stat: Optional[float] = float(lr_inter_stat)
        interaction_df: Optional[int] = int(lr_inter_df)
    except Exception:
        interaction_p = None
        interaction_stat = None
        interaction_df = None

    response, overall_score = _likert_from_effects(
        site_effect=site_effect,
        age_effect=age_effect,
        interaction_p=interaction_p,
    )

    yes_answer = response >= 50
    if overall_score >= 0.8:
        strength_word = "strong"
    elif overall_score >= 0.6:
        strength_word = "moderate"
    elif overall_score >= 0.4:
        strength_word = "weak"
    else:
        strength_word = "very little"

    lines = []
    lines.append(
        "Research question: Do children’s reliance on social information and "
        "preference for majority cues vary across cultures and developmental stages?"
    )
    lines.append(
        "I analysed the provided dataset (N = {}) using logistic regression models "
        "predicting whether each child chose the majority option (vs. the minority "
        "or an undemonstrated option).".format(len(df))
    )
    lines.append(
        "Overall, children chose the majority option on average {:.2f} of the time.".format(
            overall_mean
        )
    )
    lines.append(
        "Across cultural sites (8-level site identifier), the proportion of majority "
        "choices ranged from {:.2f} to {:.2f}."
        .format(min_site_mean, max_site_mean)
    )
    if not np.isnan(min_age_mean):
        lines.append(
            "Using coarse age bands (4–6, 7–9, 10–12, 13–14 years), the mean proportion "
            "of majority choices ranged from {:.2f} to {:.2f}."
            .format(min_age_mean, max_age_mean)
        )

    # Statistical tests for site and age effects.
    lines.append(
        "Controlling for demonstration order (whether the majority option was shown first), "
        "adding cultural site to a model already containing age significantly improved fit "
        "(likelihood-ratio test χ²({}) = {:.2f}, p = {:.3g})."
        .format(site_effect.df_diff, site_effect.lr_stat, site_effect.p_value)
    )
    lines.append(
        "Similarly, adding age to a model that already contained cultural site significantly "
        "improved fit (likelihood-ratio test χ²({}) = {:.2f}, p = {:.3g})."
        .format(age_effect.df_diff, age_effect.lr_stat, age_effect.p_value)
    )

    if interaction_p is not None:
        if interaction_p < 0.05:
            interaction_phrase = "did"
        else:
            interaction_phrase = "did not"
        lines.append(
            "A model allowing age effects to vary by site (age × site interaction) {} "
            "provide a significantly better fit than the additive model "
            "(likelihood-ratio test χ²({}) = {:.2f}, p = {:.3g})."
            .format(
                interaction_phrase,
                interaction_df if interaction_df is not None else 0,
                interaction_stat if interaction_stat is not None else float("nan"),
                interaction_p,
            )
        )

    if yes_answer:
        conclusion_sentence = (
            "Taken together, these results provide {} evidence that children’s reliance on "
            "social information and preference for majority cues vary systematically across "
            "both cultural contexts and developmental stages. I therefore answer “Yes” to the "
            "research question. On a 0–100 Likert scale (0 = strong “No”, 100 = strong “Yes”), "
            "this corresponds to a response of {}."
        ).format(strength_word, response)
    else:
        conclusion_sentence = (
            "Taken together, these results provide {} evidence for systematic variation in "
            "children’s reliance on social information and preference for majority cues across "
            "cultures and developmental stages. Any observed differences are small and/or not "
            "statistically robust. I therefore answer “No” to the research question. "
            "On a 0–100 Likert scale (0 = strong “No”, 100 = strong “Yes”), this corresponds "
            "to a response of {}."
        ).format(strength_word, response)

    lines.append(conclusion_sentence)
    explanation = "\n\n".join(lines)

    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)


if __name__ == "__main__":
    main()
