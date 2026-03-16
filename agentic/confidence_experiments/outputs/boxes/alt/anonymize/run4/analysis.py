import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Derive variables:
    # social_use: 1 if child followed any demonstrated (majority or minority) option, 0 if chose undemonstrated option
    df["social_use"] = (df["feature1"] != 1).astype(int)

    # Among socially guided choices, majority_choice: 1 if majority, 0 if minority
    df_social = df[df["social_use"] == 1].copy()
    df_social["majority_choice"] = (df_social["feature1"] == 2).astype(int)

    # Basic descriptives
    n = int(len(df))
    num_sites = int(df["feature5"].nunique())
    age_min = float(df["feature3"].min())
    age_max = float(df["feature3"].max())
    age_mean = float(df["feature3"].mean())
    age_std = float(df["feature3"].std())

    social_rate = float(df["social_use"].mean())
    majority_rate = float(df_social["majority_choice"].mean()) if len(df_social) > 0 else np.nan

    # Fit logistic models for reliance on social information
    model_social_site = smf.logit("social_use ~ feature3 + C(feature5)", data=df).fit(disp=False)
    model_social_nosite = smf.logit("social_use ~ feature3", data=df).fit(disp=False)

    age_social_p = float(model_social_site.pvalues["feature3"])
    lr_stat_social = 2.0 * (model_social_site.llf - model_social_nosite.llf)
    df_diff_social = int(model_social_site.df_model - model_social_nosite.df_model)
    site_social_p = float(chi2.sf(lr_stat_social, df_diff_social))

    beta_age_social = float(model_social_site.params["feature3"])

    # Fit logistic models for majority preference among social users
    model_maj_site = smf.logit("majority_choice ~ feature3 + C(feature5)", data=df_social).fit(disp=False)
    model_maj_nosite = smf.logit("majority_choice ~ feature3", data=df_social).fit(disp=False)

    age_majority_p = float(model_maj_site.pvalues["feature3"])
    lr_stat_maj = 2.0 * (model_maj_site.llf - model_maj_nosite.llf)
    df_diff_maj = int(model_maj_site.df_model - model_maj_nosite.df_model)
    site_majority_p = float(chi2.sf(lr_stat_maj, df_diff_maj))

    beta_age_majority = float(model_maj_site.params["feature3"])

    # Map p-values to a 0–100 response scale
    p_values = [age_social_p, site_social_p, age_majority_p, site_majority_p]
    sig_flags = [p <= 0.05 for p in p_values]
    near_sig_flags = [0.05 < p <= 0.1 for p in p_values]
    very_sig_flags = [p <= 0.001 for p in p_values]

    num_sig = int(sum(sig_flags))
    num_near_sig = int(sum(near_sig_flags))

    # Base of 10, then add weight for significant and near-significant effects
    response = 10 + 20 * num_sig + 5 * num_near_sig + 5 * sum(very_sig_flags)
    response = int(max(0, min(100, round(response))))

    # Determine overall Yes/No interpretation from the score
    yes_answer = response >= 50
    verbal_answer = "Yes" if yes_answer else "No"

    direction_social = (
        "older children showed higher reliance on social information"
        if beta_age_social > 0
        else "older children showed lower reliance on social information"
    )
    direction_majority = (
        "older children showed a stronger majority preference"
        if beta_age_majority > 0
        else "older children showed a weaker majority preference"
    )

    social_pct = social_rate * 100.0
    majority_pct = majority_rate * 100.0 if not np.isnan(majority_rate) else np.nan

    if yes_answer:
        overall_sentence = (
            "Overall, these analyses indicate that children's reliance on social information and "
            "their preference for majority cues do vary across ages and cultural sites in this dataset."
        )
    else:
        overall_sentence = (
            "Overall, these analyses provide limited statistical evidence that children's reliance on "
            "social information or their preference for majority cues differ systematically across "
            "ages or cultural sites in this sample."
        )

    explanation = (
        f"The dataset contains {n} children aged {age_min:.0f}–{age_max:.0f} years "
        f"(mean {age_mean:.2f}, SD {age_std:.2f}) sampled from {num_sites} cultural sites "
        f"(feature5). The outcome feature (feature1) encodes whether each child chose the "
        f"undemonstrated option (1), the majority option (2), or the minority option (3). "
        f"I defined two derived measures: (a) reliance on social information, coded as 1 when "
        f"children followed any demonstrated option (majority or minority) versus 0 when they "
        f"chose an undemonstrated option, and (b) majority preference among socially guided "
        f"choices, coded as 1 when children chose the majority option versus 0 when they chose "
        f"the minority option.\n\n"
        f"Descriptively, {social_pct:.1f}% of children relied on social information by choosing "
        f"either the majority or minority option. Among those who used social information, "
        f"{majority_pct:.1f}% chose the majority option, indicating an overall bias toward the "
        f"majority cue.\n\n"
        f"To assess developmental and cultural variation, I fit logistic regression models for "
        f"each derived measure with age in years (feature3) as a continuous predictor and site "
        f"(feature5) entered as a categorical predictor. For reliance on social information, the "
        f"age coefficient had p = {age_social_p:.3g} and the likelihood-ratio test for site yielded "
        f"p = {site_social_p:.3g}. For majority preference among children who used social "
        f"information, the corresponding p-values were p = {age_majority_p:.3g} for age and "
        f"p = {site_majority_p:.3g} for site. At a conventional α = 0.05 threshold, {num_sig} of "
        f"these four tests are statistically significant and {num_near_sig} additional tests are "
        f"near-significant (0.05 < p ≤ 0.10). Although point estimates suggest that {direction_social} "
        f"and that {direction_majority}, these patterns should be interpreted cautiously given the "
        f"p-values.\n\n"
        f"{overall_sentence} Accordingly, I give a '{verbal_answer}' answer to the research "
        f"question. The numerical response value of {response} on the 0–100 scale reflects the "
        f"strength of evidence in favor of such developmental and cross-cultural variation, with "
        f"higher values indicating stronger support."
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
