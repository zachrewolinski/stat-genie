import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def likelihood_ratio_pvalue(full_model, reduced_model):
    """Compute likelihood-ratio test p-value comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    if df_diff <= 0:
        return 1.0
    return float(chi2.sf(lr_stat, df_diff))


def map_p_to_likert(min_p):
    """
    Map the strength of evidence for variation (smallest p-value)
    to an integer Likert score in [-100, 100], where positive values
    indicate evidence that children's reliance on social information
    and majority preference DO vary across age and cultures.
    """
    if min_p < 1e-8:
        return 100
    if min_p < 1e-6:
        return 95
    if min_p < 1e-4:
        return 90
    if min_p < 1e-3:
        return 80
    if min_p < 1e-2:
        return 60
    if min_p < 0.05:
        return 40
    if min_p < 0.1:
        return 20
    if min_p < 0.2:
        return 0
    if min_p < 0.5:
        return -20
    return -40


def main():
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Social information use: 1 if child followed any demonstrated option (majority or minority)
    df["is_social"] = df["y"].isin([2, 3]).astype(int)

    # Majority preference among those who used social information
    df["is_majority"] = (df["y"] == 2).astype(int)

    # Model 1: Does reliance on social information vary by age and culture?
    social_full = smf.logit("is_social ~ age + C(culture)", data=df).fit(disp=False)
    social_age_only = smf.logit("is_social ~ age", data=df).fit(disp=False)
    social_culture_only = smf.logit("is_social ~ C(culture)", data=df).fit(disp=False)

    p_culture_social = likelihood_ratio_pvalue(social_full, social_age_only)
    p_age_social = likelihood_ratio_pvalue(social_full, social_culture_only)

    # Model 2: Among social learners, does majority preference vary by age and culture?
    df_social = df[df["is_social"] == 1].copy()
    majority_full = smf.logit("is_majority ~ age + C(culture)", data=df_social).fit(disp=False)
    majority_age_only = smf.logit("is_majority ~ age", data=df_social).fit(disp=False)
    majority_culture_only = smf.logit("is_majority ~ C(culture)", data=df_social).fit(disp=False)

    p_culture_majority = likelihood_ratio_pvalue(majority_full, majority_age_only)
    p_age_majority = likelihood_ratio_pvalue(majority_full, majority_culture_only)

    # Combine evidence: take the smallest p-value across tests
    p_values = [
        p_culture_social,
        p_age_social,
        p_culture_majority,
        p_age_majority,
    ]
    min_p = float(np.nanmin(p_values))

    likert_score = map_p_to_likert(min_p)

    # Write ONLY the scalar to conclusion.txt, as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(likert_score)))


if __name__ == "__main__":
    main()

