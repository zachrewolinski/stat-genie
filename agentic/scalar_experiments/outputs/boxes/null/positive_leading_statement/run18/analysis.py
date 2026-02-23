import json
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing two nested models."""
    lr_stat = 2.0 * (full_model.llf - reduced_model.llf)
    df_diff = int(full_model.df_model - reduced_model.df_model)
    p_value = float(chi2.sf(lr_stat, df_diff))
    return float(lr_stat), df_diff, p_value


def strength_from_p_and_range(p_value, value_range):
    """
    Convert a p-value and range of proportions into a 0–1 strength score.

    - Smaller p-values and larger ranges yield values closer to 1.
    - Uses -log10(p) for significance and scales range relative to 0.5.
    """
    # Guard against p = 0 numerically
    p_safe = max(min(p_value, 1.0), 1e-300)
    sig_raw = -np.log10(p_safe)
    sig_norm = float(np.clip(sig_raw / 10.0, 0.0, 1.0))

    eff_norm = float(np.clip(value_range / 0.5, 0.0, 1.0))
    return sig_norm, eff_norm


def main():
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["social_choice"] = (df["y"] != 1).astype(int)
    df["age_c"] = df["age"] - df["age"].mean()

    n = len(df)
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
    n_cultures = int(df["culture"].nunique())

    overall_social = float(df["social_choice"].mean())
    overall_majority = float(df["majority_choice"].mean())

    # --- Models for majority choice ---
    formula_full_major = "majority_choice ~ age_c + C(culture) + gender + majority_first"
    formula_no_culture_major = "majority_choice ~ age_c + gender + majority_first"
    formula_no_age_major = "majority_choice ~ C(culture) + gender + majority_first"

    model_full_major = smf.glm(
        formula=formula_full_major,
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_no_culture_major = smf.glm(
        formula=formula_no_culture_major,
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_no_age_major = smf.glm(
        formula=formula_no_age_major,
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    lr_culture_major, df_culture_major, p_culture_major = lr_test(
        model_full_major, model_no_culture_major
    )
    lr_age_major, df_age_major, p_age_major = lr_test(
        model_full_major, model_no_age_major
    )

    majority_rates_by_culture = df.groupby("culture")["majority_choice"].mean()
    majority_rates_by_age = df.groupby("age")["majority_choice"].mean()
    range_culture_major = float(
        majority_rates_by_culture.max() - majority_rates_by_culture.min()
    )
    range_age_major = float(
        majority_rates_by_age.max() - majority_rates_by_age.min()
    )

    # --- Models for general social-information use ---
    formula_full_social = "social_choice ~ age_c + C(culture) + gender + majority_first"
    formula_no_culture_social = "social_choice ~ age_c + gender + majority_first"
    formula_no_age_social = "social_choice ~ C(culture) + gender + majority_first"

    model_full_social = smf.glm(
        formula=formula_full_social,
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_no_culture_social = smf.glm(
        formula=formula_no_culture_social,
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_no_age_social = smf.glm(
        formula=formula_no_age_social,
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    lr_culture_social, df_culture_social, p_culture_social = lr_test(
        model_full_social, model_no_culture_social
    )
    lr_age_social, df_age_social, p_age_social = lr_test(
        model_full_social, model_no_age_social
    )

    social_rates_by_culture = df.groupby("culture")["social_choice"].mean()
    social_rates_by_age = df.groupby("age")["social_choice"].mean()
    range_culture_social = float(
        social_rates_by_culture.max() - social_rates_by_culture.min()
    )
    range_age_social = float(
        social_rates_by_age.max() - social_rates_by_age.min()
    )

    # --- Compute an overall strength score for the Yes/No scale ---
    strength_components = []
    for p_val, val_range in [
        (p_culture_major, range_culture_major),
        (p_age_major, range_age_major),
        (p_culture_social, range_culture_social),
        (p_age_social, range_age_social),
    ]:
        sig_norm, eff_norm = strength_from_p_and_range(p_val, val_range)
        strength_components.extend([sig_norm, eff_norm])

    overall_strength = float(np.mean(strength_components)) if strength_components else 0.0

    # Evidence for "Yes" if age or culture terms are significant
    yes_evidence = any(
        [
            (p_culture_major < 0.05 and range_culture_major > 0.1),
            (p_age_major < 0.05 and range_age_major > 0.1),
            (p_culture_social < 0.05 and range_culture_social > 0.05),
            (p_age_social < 0.05 and range_age_social > 0.05),
        ]
    )

    if yes_evidence:
        response = 50 + int(round(50.0 * overall_strength))
    else:
        response = 50 - int(round(50.0 * overall_strength))

    response = int(np.clip(response, 0, 100))

    base_explanation = dedent(
        f"""
        Research question
        -----------------
        Do children's reliance on social information and preference for majority cues
        vary across cultures and developmental stages?

        Data and operationalisation
        ---------------------------
        - N = {n} children aged {age_min}–{age_max} years from {n_cultures} cultural sites (coded 1–{n_cultures}).
        - Outcome y was coded as 1 = undemonstrated option, 2 = majority option,
          and 3 = minority option.
        - I defined "social-information use" as choosing any demonstrated option
          (y = 2 or 3) versus the undemonstrated option (y = 1), and "majority
          preference" as choosing the majority option (y = 2) versus the other two.
        - Overall, children chose a demonstrated option on {overall_social * 100:.1f}% of trials
          and the majority option on {overall_majority * 100:.1f}% of trials, indicating
          robust social learning and a majority bias on average.

        Statistical analysis
        --------------------
        - I fit logistic regression models (binomial GLMs) predicting majority choice
          and social-information use from age (centered), culture (categorical),
          gender, and whether the majority option was demonstrated first.
        - To test for developmental and cultural variation, I used likelihood-ratio
          tests comparing full models with age and culture terms to reduced models
          without those predictors.

        Key results
        -----------
        Majority preference (choosing the majority option):
        - Cultural differences: LR(df = {df_culture_major}) = {lr_culture_major:.2f}, p = {p_culture_major:.3g}.
          The proportion of majority choices across sites ranged from
          {majority_rates_by_culture.min() * 100:.1f}% to {majority_rates_by_culture.max() * 100:.1f}%
          (range = {range_culture_major * 100:.1f} percentage points).
        - Developmental change: LR(df = {df_age_major}) = {lr_age_major:.2f}, p = {p_age_major:.3g}.
          Across single-year ages, majority-choice rates ranged from
          {majority_rates_by_age.min() * 100:.1f}% to {majority_rates_by_age.max() * 100:.1f}%
          (range = {range_age_major * 100:.1f} percentage points).

        Social-information use (choosing any demonstrated option):
        - Cultural differences: LR(df = {df_culture_social}) = {lr_culture_social:.2f}, p = {p_culture_social:.3g}.
          Site-level social-choice rates spanned
          {social_rates_by_culture.min() * 100:.1f}%–{social_rates_by_culture.max() * 100:.1f}%
          (range = {range_culture_social * 100:.1f} percentage points).
        - Developmental change: LR(df = {df_age_social}) = {lr_age_social:.2f}, p = {p_age_social:.3g}.
          Age-specific social-choice rates ranged from
          {social_rates_by_age.min() * 100:.1f}% to {social_rates_by_age.max() * 100:.1f}%
          (range = {range_age_social * 100:.1f} percentage points).
        """
    ).strip()

    if yes_evidence:
        interpretation_lines = [
            "Interpretation",
            "--------------",
            "- The likelihood-ratio tests provide statistically reliable evidence that",
            "  at least some aspects of children's majority preference and/or general",
            "  social-information use differ across cultures and/or change with age",
            "  (with p-values below conventional 0.05 thresholds).",
            "- The observed differences in choice proportions across sites and ages are",
            "  practically meaningful, indicating that children's reliance on social",
            "  information and majority cues is not invariant but varies with context",
            "  and development.",
            "",
            "Conclusion",
            "----------",
            "Overall, the pattern of results supports a \"Yes\" answer to the research",
            "question: children's reliance on social information and their preference",
            "for majority cues vary across cultures and developmental stages.",
        ]
        interpretation_block = "\n".join(interpretation_lines)
    else:
        interpretation_lines = [
            "Interpretation",
            "--------------",
            "- The likelihood-ratio tests do not provide strong statistical evidence",
            "  that majority preference or general social-information use differ across",
            "  cultures or change systematically with age: all tested age and culture",
            "  terms have p-values greater than 0.05.",
            (
                "- Although raw choice proportions differ by up to "
                f"{range_culture_major * 100:.1f} percentage points across sites and "
                f"{range_age_major * 100:.1f} percentage points across ages for majority"
            ),
            "  preference (and similarly for social-information use), these differences",
            "  are not statistically reliable given the sample sizes and variability.",
            "- Taken together, the data are more consistent with modest or noisy",
            "  variation than with clear, robust cross-cultural or developmental",
            "  differences in social-information use or majority preference.",
            "",
            "Conclusion",
            "----------",
            "Based on this dataset alone, there is insufficient statistical evidence to",
            "conclude that children's reliance on social information and preference for",
            "majority cues vary reliably across cultures and developmental stages. In",
            "line with best statistical practice, this supports a more cautious, largely",
            "\"No\" (or at least inconclusive) answer to the research question.",
        ]
        interpretation_block = "\n".join(interpretation_lines)

    explanation = f"{base_explanation}\n\n{interpretation_block}"

    result = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
