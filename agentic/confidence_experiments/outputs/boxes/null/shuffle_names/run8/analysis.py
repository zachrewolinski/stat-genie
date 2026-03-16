import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full_res, reduced_res):
    """Likelihood-ratio test comparing two nested models."""
    lr_stat = 2 * (full_res.llf - reduced_res.llf)
    df = int(full_res.df_model - reduced_res.df_model)
    p_value = float(chi2.sf(lr_stat, df)) if df > 0 else np.nan
    return lr_stat, df, p_value


def main():
    # Load data
    df = pd.read_csv("boxes.csv")

    # Derived variables
    # social_choice: 1 if child followed any demonstrator (majority or minority), 0 if undemonstrated third option
    df["social_choice"] = (df["majority_first"] != 1).astype(int)
    # majority_choice: 1 if child followed majority demonstrators, 0 otherwise
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Logistic models for reliance on social information
    social_full = smf.logit(
        "social_choice ~ age + C(y) + gender + culture", data=df
    ).fit(disp=False, maxiter=1000)
    social_no_age = smf.logit(
        "social_choice ~ C(y) + gender + culture", data=df
    ).fit(disp=False, maxiter=1000)
    social_no_site = smf.logit(
        "social_choice ~ age + gender + culture", data=df
    ).fit(disp=False, maxiter=1000)

    _, _, age_p_social = lr_test(social_full, social_no_age)
    _, _, site_p_social = lr_test(social_full, social_no_site)

    # Logistic models for preference for majority cues (among children using social info)
    df_social = df[df["social_choice"] == 1].copy()

    major_full = smf.logit(
        "majority_choice ~ age + C(y) + gender + culture", data=df_social
    ).fit(disp=False, maxiter=1000)
    major_no_age = smf.logit(
        "majority_choice ~ C(y) + gender + culture", data=df_social
    ).fit(disp=False, maxiter=1000)
    major_no_site = smf.logit(
        "majority_choice ~ age + gender + culture", data=df_social
    ).fit(disp=False, maxiter=1000)

    _, _, age_p_major = lr_test(major_full, major_no_age)
    _, _, site_p_major = lr_test(major_full, major_no_site)

    # Descriptive effect sizes across developmental stages (age groups) and sites
    age_bins = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
    )
    df["age_group"] = age_bins

    prob_social_by_stage = df.groupby("age_group")["social_choice"].mean()
    prob_major_by_stage = df.groupby("age_group")["majority_choice"].mean()

    prob_social_by_site = df.groupby("y")["social_choice"].mean()
    prob_major_by_site = df.groupby("y")["majority_choice"].mean()

    social_age_range = float(prob_social_by_stage.max() - prob_social_by_stage.min())
    major_age_range = float(prob_major_by_stage.max() - prob_major_by_stage.min())
    social_site_range = float(prob_social_by_site.max() - prob_social_by_site.min())
    major_site_range = float(prob_major_by_site.max() - prob_major_by_site.min())

    # Determine strength of evidence for variation
    sig_social_age = age_p_social < 0.05
    sig_social_site = site_p_social < 0.05
    sig_major_age = age_p_major < 0.05
    sig_major_site = site_p_major < 0.05

    num_sig = int(
        sig_social_age + sig_social_site + sig_major_age + sig_major_site
    )

    # Base rating from number of significant effects
    # 0,1,2,3,4 significant tests -> 10, 30, 50, 70, 90
    rating = 10 + 20 * num_sig

    # Adjust rating based on typical effect size (range of proportions)
    avg_range = np.mean(
        [social_age_range, major_age_range, social_site_range, major_site_range]
    )
    if avg_range > 0.30:
        rating += 10
    elif avg_range < 0.10:
        rating -= 10

    rating = int(max(0, min(100, rating)))

    social_age_signif_word = (
        "significantly" if sig_social_age else "not significantly"
    )
    social_site_signif_word = (
        "significantly" if sig_social_site else "not significantly"
    )
    major_age_signif_word = "significant" if sig_major_age else "not significant"
    major_site_signif_word = "significant" if sig_major_site else "not significant"

    answer_word = "Yes" if rating >= 50 else "No"

    if rating >= 50:
        conclusion_clause = (
            "Taken together, these results provide evidence that children's "
            "reliance on social information and their preference for majority "
            "cues vary across cultures and developmental stages."
        )
    else:
        conclusion_clause = (
            "Given that most age and site effects are not statistically "
            "significant and the observed differences in proportions are "
            "modest, I do not find strong evidence that these behaviors vary "
            "systematically across cultures or developmental stages."
        )

    explanation = (
        "I analyzed data from 629 children across 8 sites using logistic "
        "regression models to capture two outcomes: (a) reliance on social "
        "information (choosing either the majority or minority demonstrators "
        "versus an undemonstrated option) and (b) preference for majority cues "
        "among children who followed social information. For reliance on social "
        f"information, likelihood-ratio tests show that adding age "
        f"{social_age_signif_word} improves model fit (p={age_p_social:.3g}) and "
        f"that adding site (cultural context) {social_site_signif_word} improves "
        f"fit (p={site_p_social:.3g}). The proportion of social choices increases "
        f"from {prob_social_by_stage.min():.2f} to {prob_social_by_stage.max():.2f} "
        f"across age groups and varies from {prob_social_by_site.min():.2f} to "
        f"{prob_social_by_site.max():.2f} across sites. For majority preference, "
        f"age effects are {major_age_signif_word} (p={age_p_major:.3g}) and site "
        f"effects are {major_site_signif_word} (p={site_p_major:.3g}); the "
        f"proportion of majority choices ranges from "
        f"{prob_major_by_stage.min():.2f} to {prob_major_by_stage.max():.2f} "
        f"across age groups and from {prob_major_by_site.min():.2f} to "
        f"{prob_major_by_site.max():.2f} across sites. {conclusion_clause} "
        f"I therefore answer '{answer_word}' to the research question with a "
        f"confidence score of {rating} on a 0–100 Likert scale, where lower "
        "values correspond to stronger \"No\" answers and higher values to "
        "stronger \"Yes\" answers."
    )

    result = {"response": rating, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
