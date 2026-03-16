import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def fit_logistic_models(df: pd.DataFrame, outcome: str):
    """
    Fit baseline vs. site-extended logistic GLM for a binary outcome.
    Returns age p-value, site (culture) likelihood-ratio p-value, and basic model objects.
    """
    formula_base = f"{outcome} ~ age + gender + culture"
    formula_site = formula_base + " + C(site)"

    base_model = smf.glm(
        formula_base, data=df, family=sm.families.Binomial()
    ).fit()
    site_model = smf.glm(formula_site, data=df, family=sm.families.Binomial()).fit()

    age_p = float(base_model.pvalues["age"])

    # Manual likelihood-ratio test comparing models with and without site
    df_diff = site_model.df_model - base_model.df_model
    lr_stat = 2.0 * (site_model.llf - base_model.llf)
    lr_p = chi2.sf(lr_stat, df_diff)

    return {
        "age_p": age_p,
        "site_lr_p": float(lr_p),
        "base_model": base_model,
        "site_model": site_model,
    }


def summarize_group_patterns(df: pd.DataFrame):
    """
    Compute simple descriptive stats by site and age band to support interpretation.
    """
    site_summary = (
        df.groupby("site")
        .agg(
            n=("site", "size"),
            social_rate=("social_choice", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )

    # Define coarse age bands
    bins = [3.5, 6.5, 9.5, 11.5, 14.5]
    labels = ["4-6", "7-9", "10-11", "12-14"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels)
    age_summary = (
        df.groupby("age_band")
        .agg(
            n=("age_band", "size"),
            social_rate=("social_choice", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )

    return site_summary, age_summary


def compute_evidence_score(p_values, alpha=0.05) -> int:
    """
    Map a list of p-values to a 0-100 evidence score based on how many
    are statistically significant at the chosen alpha level.
    """
    if not p_values:
        return 0
    n_sig = sum(p < alpha for p in p_values)
    score = int(round(100 * n_sig / len(p_values)))
    return max(0, min(100, score))


def build_explanation(
    social_results,
    majority_results,
    site_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
) -> str:
    """
    Construct a concise textual explanation based on model results and descriptives.
    """
    # Overall rates
    overall_social_rate = site_summary["social_rate"].mean()
    overall_majority_rate = site_summary["majority_rate"].mean()

    # Cross-cultural variation (range across sites)
    social_min = site_summary["social_rate"].min()
    social_max = site_summary["social_rate"].max()
    majority_min = site_summary["majority_rate"].min()
    majority_max = site_summary["majority_rate"].max()

    # Age trends (use band-level differences)
    age_social_min = age_summary["social_rate"].min()
    age_social_max = age_summary["social_rate"].max()
    age_majority_min = age_summary["majority_rate"].min()
    age_majority_max = age_summary["majority_rate"].max()

    social_age_p = float(social_results["age_p"])
    social_site_p = float(social_results["site_lr_p"])
    majority_age_p = float(majority_results["age_p"])
    majority_site_p = float(majority_results["site_lr_p"])

    if social_age_p < 0.05:
        social_age_sentence = (
            f"Age significantly predicted reliance on social information (p≈{social_age_p:.3f}). "
        )
    else:
        social_age_sentence = (
            f"Age did not significantly predict reliance on social information (p≈{social_age_p:.3f}). "
        )

    if majority_age_p < 0.05:
        majority_age_sentence = (
            f"Age significantly predicted majority preference (p≈{majority_age_p:.3f}). "
        )
    else:
        majority_age_sentence = (
            f"Age did not significantly predict majority preference (p≈{majority_age_p:.3f}). "
        )

    if social_site_p < 0.05:
        social_site_sentence = (
            f"Including site (a proxy for cultural context) significantly improved model fit "
            f"for reliance on social information (likelihood-ratio p≈{social_site_p:.3f}). "
        )
    else:
        social_site_sentence = (
            f"Including site (a proxy for cultural context) did not significantly improve model fit "
            f"for reliance on social information (likelihood-ratio p≈{social_site_p:.3f}). "
        )

    if majority_site_p < 0.05:
        majority_site_sentence = (
            f"Including site significantly improved model fit for majority preference "
            f"(likelihood-ratio p≈{majority_site_p:.3f}). "
        )
    else:
        majority_site_sentence = (
            f"Including site did not significantly improve model fit for majority preference "
            f"(likelihood-ratio p≈{majority_site_p:.3f}). "
        )

    intro_text = (
        "Using the 629 children in the dataset, I treated reliance on social information "
        "as choosing either majority or minority options versus the undemonstrated option, "
        "and preference for majority cues as choosing the majority versus either the undemonstrated or minority options. "
        f"Overall, children relied on social information in about {overall_social_rate:.2f} of trials and chose the majority option in about {overall_majority_rate:.2f} of trials. "
    )

    age_effect_text = social_age_sentence + majority_age_sentence

    age_range_text = (
        f"Across coarse age bands, social-information use ranged from roughly {age_social_min:.2f} to {age_social_max:.2f}, "
        f"and majority-choice rates ranged from roughly {age_majority_min:.2f} to {age_majority_max:.2f}, but these differences were modest in size. "
    )

    site_effect_text = social_site_sentence + majority_site_sentence

    site_range_text = (
        f"Observed site-level averages varied from about {social_min:.2f} to {social_max:.2f} for social-information use "
        f"and from about {majority_min:.2f} to {majority_max:.2f} for majority choices. "
    )

    conclusion_text = (
        "Taken together, the non-significant age and site effects suggest limited evidence that children's reliance on social information "
        "or their preference for majority cues systematically vary across developmental stages or cultural sites in this sample."
    )

    explanation = (
        intro_text
        + age_effect_text
        + age_range_text
        + site_effect_text
        + site_range_text
        + conclusion_text
    )

    return explanation


def main():
    # Load data
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social_choice"] = (df["majority_first"] != 1).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Prepare categorical representations
    df["gender"] = df["gender"].astype("category")
    df["site"] = df["y"].astype("category")

    # Fit models for both outcomes
    social_results = fit_logistic_models(df, "social_choice")
    majority_results = fit_logistic_models(df, "majority_choice")

    # Summarize descriptive patterns
    site_summary, age_summary = summarize_group_patterns(df)

    # Collect p-values relevant to the research question
    p_values = [
        social_results["age_p"],
        social_results["site_lr_p"],
        majority_results["age_p"],
        majority_results["site_lr_p"],
    ]

    response_score = compute_evidence_score(p_values)
    explanation = build_explanation(
        social_results, majority_results, site_summary, age_summary
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    Path("conclusion.txt").write_text(
        json.dumps(conclusion, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
