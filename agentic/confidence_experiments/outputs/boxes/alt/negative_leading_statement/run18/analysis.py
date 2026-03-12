import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def fit_logit(formula: str, data: pd.DataFrame):
    """Fit a logistic regression model, returning the fitted results."""
    model = smf.logit(formula, data=data)
    result = model.fit(disp=0, maxiter=1000)
    return result


def lr_test(full_model, reduced_model):
    """Likelihood ratio test between two nested models; return (stat, pval, df)."""
    try:
        stat = 2.0 * (full_model.llf - reduced_model.llf)
        df = int(round(full_model.df_model - reduced_model.df_model))
        if df <= 0 or not np.isfinite(stat):
            return np.nan, np.nan, df
        pval = chi2.sf(stat, df)
        return float(stat), float(pval), df
    except Exception:
        return np.nan, np.nan, 0


def evidence_from_p(p: float) -> float:
    """Convert a p-value into an evidence score between 0 and 1."""
    if not np.isfinite(p):
        return 0.0
    if p < 0.001:
        return 1.0
    if p < 0.01:
        return 0.8
    if p < 0.05:
        return 0.6
    if p < 0.1:
        return 0.3
    return 0.0


def magnitude_score(diff: float) -> float:
    """Convert a probability difference (0-1) into a 0-1 score."""
    diff = float(abs(diff))
    return float(np.clip(diff / 0.5, 0.0, 1.0))


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Derived variables
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["demonstrated_choice"] = df["y"].isin([2, 3])

    n_total = len(df)

    # Age groups for descriptive summaries
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
    )

    # Descriptive statistics: reliance on social information
    social_overall = df["social_choice"].mean()
    social_by_culture = df.groupby("culture")["social_choice"].mean()
    social_by_age = df.groupby("age_group")["social_choice"].mean()

    # Descriptive statistics: majority preference (among demonstrated choices)
    df_demo = df[df["demonstrated_choice"]].copy()
    n_demo = len(df_demo)
    majority_overall = df_demo["majority_choice"].mean() if n_demo > 0 else np.nan
    majority_by_culture = (
        df_demo.groupby("culture")["majority_choice"].mean() if n_demo > 0 else None
    )
    majority_by_age = (
        df_demo.groupby("age_group")["majority_choice"].mean() if n_demo > 0 else None
    )

    # Logistic regression: reliance on social information
    social_full = fit_logit(
        "social_choice ~ age + C(culture) + gender + majority_first", df
    )
    social_no_culture = fit_logit(
        "social_choice ~ age + gender + majority_first", df
    )
    social_no_age = fit_logit(
        "social_choice ~ C(culture) + gender + majority_first", df
    )

    social_lr_culture_stat, social_lr_culture_p, social_lr_culture_df = lr_test(
        social_full, social_no_culture
    )
    social_lr_age_stat, social_lr_age_p, social_lr_age_df = lr_test(
        social_full, social_no_age
    )

    # Logistic regression: majority preference among demonstrated choices
    maj_lr_culture_stat = maj_lr_culture_p = maj_lr_culture_df = np.nan
    maj_lr_age_stat = maj_lr_age_p = maj_lr_age_df = np.nan

    if n_demo > 0:
        maj_full = fit_logit(
            "majority_choice ~ age + C(culture) + gender + majority_first", df_demo
        )
        maj_no_culture = fit_logit(
            "majority_choice ~ age + gender + majority_first", df_demo
        )
        maj_no_age = fit_logit(
            "majority_choice ~ C(culture) + gender + majority_first", df_demo
        )

        maj_lr_culture_stat, maj_lr_culture_p, maj_lr_culture_df = lr_test(
            maj_full, maj_no_culture
        )
        maj_lr_age_stat, maj_lr_age_p, maj_lr_age_df = lr_test(maj_full, maj_no_age)

    # Evidence scores for variation across cultures and ages
    components = []

    # Social information – culture
    social_culture_sig = evidence_from_p(social_lr_culture_p)
    social_culture_diff = (
        (social_by_culture.max() - social_by_culture.min())
        if len(social_by_culture) > 0
        else 0.0
    )
    social_culture_mag = magnitude_score(social_culture_diff) if social_lr_culture_p < 0.05 else 0.0
    components.append(0.7 * social_culture_sig + 0.3 * social_culture_mag)

    # Social information – age
    social_age_sig = evidence_from_p(social_lr_age_p)
    social_age_diff = (
        (social_by_age.max() - social_by_age.min())
        if len(social_by_age) > 0
        else 0.0
    )
    social_age_mag = magnitude_score(social_age_diff) if social_lr_age_p < 0.05 else 0.0
    components.append(0.7 * social_age_sig + 0.3 * social_age_mag)

    # Majority preference – culture
    if majority_by_culture is not None:
        maj_culture_sig = evidence_from_p(maj_lr_culture_p)
        maj_culture_diff = (
            (majority_by_culture.max() - majority_by_culture.min())
            if len(majority_by_culture) > 0
            else 0.0
        )
        maj_culture_mag = (
            magnitude_score(maj_culture_diff) if maj_lr_culture_p < 0.05 else 0.0
        )
        components.append(0.7 * maj_culture_sig + 0.3 * maj_culture_mag)

    # Majority preference – age
    if majority_by_age is not None:
        maj_age_sig = evidence_from_p(maj_lr_age_p)
        maj_age_diff = (
            (majority_by_age.max() - majority_by_age.min())
            if len(majority_by_age) > 0
            else 0.0
        )
        maj_age_mag = magnitude_score(maj_age_diff) if maj_lr_age_p < 0.05 else 0.0
        components.append(0.7 * maj_age_sig + 0.3 * maj_age_mag)

    if components:
        overall_evidence = float(np.clip(np.mean(components), 0.0, 1.0))
    else:
        overall_evidence = 0.0

    likert_response = int(round(overall_evidence * 100))

    # Build explanation text
    explanation_lines = []
    explanation_lines.append(
        "Research question: Do children's reliance on social information "
        "and preference for majority cues vary across cultures and developmental stages?"
    )
    explanation_lines.append(
        f"Data: N = {n_total} children from 8 cultural sites, ages 4–14 years."
    )
    explanation_lines.append(
        f"Reliance on social information (choosing either majority or minority demonstrator): "
        f"overall {social_overall:.1%} of choices followed a demonstrated option."
    )
    explanation_lines.append(
        f"Across cultures, the proportion of social choices ranged from "
        f"{social_by_culture.min():.1%} to {social_by_culture.max():.1%}."
    )
    explanation_lines.append(
        f"Across age groups, social choices ranged from "
        f"{social_by_age.min():.1%} to {social_by_age.max():.1%}."
    )
    explanation_lines.append(
        "Logistic regression for social information (social vs undemonstrated) "
        "included age, culture, gender, and whether the majority was demonstrated first."
    )
    explanation_lines.append(
        f"Culture effect on social information: LR χ²({social_lr_culture_df}) = "
        f"{social_lr_culture_stat:.2f}, p = {social_lr_culture_p:.3g}."
    )
    explanation_lines.append(
        f"Age effect on social information: LR χ²({social_lr_age_df}) = "
        f"{social_lr_age_stat:.2f}, p = {social_lr_age_p:.3g}."
    )

    explanation_lines.append(
        f"Preference for the majority among children who chose a demonstrated option "
        f"(N = {n_demo}): overall {majority_overall:.1%} selected the majority option."
    )
    if majority_by_culture is not None:
        explanation_lines.append(
            f"Across cultures, majority choices ranged from "
            f"{majority_by_culture.min():.1%} to {majority_by_culture.max():.1%}."
        )
    if majority_by_age is not None:
        explanation_lines.append(
            f"Across age groups, majority choices ranged from "
            f"{majority_by_age.min():.1%} to {majority_by_age.max():.1%}."
        )
    explanation_lines.append(
        "Logistic regression for majority preference (majority vs minority) "
        "again included age, culture, gender, and majority-first."
    )
    explanation_lines.append(
        f"Culture effect on majority preference: LR χ²({maj_lr_culture_df}) = "
        f"{maj_lr_culture_stat:.2f}, p = {maj_lr_culture_p:.3g}."
    )
    explanation_lines.append(
        f"Age effect on majority preference: LR χ²({maj_lr_age_df}) = "
        f"{maj_lr_age_stat:.2f}, p = {maj_lr_age_p:.3g}."
    )

    if likert_response >= 70:
        qualitative = (
            "These analyses provide strong evidence that both reliance on social "
            "information and majority preference vary meaningfully across cultures "
            "and across age (developmental stage)."
        )
    elif likert_response >= 40:
        qualitative = (
            "The analyses suggest some evidence of variation in children's reliance "
            "on social information and/or majority preference across cultures and age, "
            "but the effects are moderate or uneven across outcomes."
        )
    else:
        qualitative = (
            "Overall, the analyses do not provide strong evidence that children's "
            "reliance on social information or preference for majority cues vary "
            "substantially across cultures or developmental stages in this dataset."
        )

    explanation_lines.append(qualitative)
    explanation_lines.append(
        f"The Likert-scale response of {likert_response} (0 = strong 'No', 100 = strong 'Yes') "
        "summarizes the overall strength of evidence that such variation exists."
    )

    explanation = "\n".join(explanation_lines)

    output = {
        "response": likert_response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
