from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def contribution(p_value, delta):
    """Convert a p-value and effect size into a scalar contribution."""
    score = 0

    if np.isfinite(p_value):
        if p_value < 0.001:
            score += 40
        elif p_value < 0.01:
            score += 30
        elif p_value < 0.05:
            score += 20
        elif p_value < 0.1:
            score += 10
        else:
            score -= 10

    if np.isfinite(delta):
        if delta > 0.25:
            score += 20
        elif delta > 0.15:
            score += 10
        elif delta < 0.05:
            score -= 10

    return score


def main():
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Encode key social-learning outcomes
    # follow_social: chose any demonstrated option (majority or minority)
    df["follow_social"] = (df["y"] != 1).astype(int)
    # choose_majority: specifically chose the majority option
    df["choose_majority"] = (df["y"] == 2).astype(int)

    social_mask = df["follow_social"] == 1
    majority_df = df.loc[social_mask].copy()
    # Among social followers, distinguish majority vs minority choice
    majority_df["majority_vs_minority"] = (majority_df["y"] == 2).astype(int)

    # GLM (binomial) models for reliance on social information
    social_full = smf.glm(
        "follow_social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    social_no_age = smf.glm(
        "follow_social ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    social_no_culture = smf.glm(
        "follow_social ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    _, _, p_age_social = lr_test(social_full, social_no_age)
    _, _, p_culture_social = lr_test(social_full, social_no_culture)

    # GLM (binomial) models for majority preference among social learners
    majority_full = smf.glm(
        "majority_vs_minority ~ age + C(culture)",
        data=majority_df,
        family=sm.families.Binomial(),
    ).fit()
    majority_no_age = smf.glm(
        "majority_vs_minority ~ C(culture)",
        data=majority_df,
        family=sm.families.Binomial(),
    ).fit()
    majority_no_culture = smf.glm(
        "majority_vs_minority ~ age",
        data=majority_df,
        family=sm.families.Binomial(),
    ).fit()

    _, _, p_age_majority = lr_test(majority_full, majority_no_age)
    _, _, p_culture_majority = lr_test(majority_full, majority_no_culture)

    # Effect sizes: variation across cultures
    prop_social_by_culture = df.groupby("culture")["follow_social"].mean()
    delta_culture_social = prop_social_by_culture.max() - prop_social_by_culture.min()

    prop_majority_by_culture = df.groupby("culture")["choose_majority"].mean()
    delta_culture_majority = (
        prop_majority_by_culture.max() - prop_majority_by_culture.min()
    )

    delta_culture_avg = float(
        np.mean([delta_culture_social, delta_culture_majority])
    )

    # Effect sizes: variation across developmental stages (age groups)
    df["age_group"] = pd.qcut(df["age"], q=3, duplicates="drop")

    prop_social_by_agegroup = df.groupby("age_group")["follow_social"].mean()
    delta_age_social = prop_social_by_agegroup.max() - prop_social_by_agegroup.min()

    prop_majority_by_agegroup = df.groupby("age_group")["choose_majority"].mean()
    delta_age_majority = (
        prop_majority_by_agegroup.max() - prop_majority_by_agegroup.min()
    )

    delta_age_avg = float(np.mean([delta_age_social, delta_age_majority]))

    # Combine evidence across social reliance and majority preference
    p_culture_avg = float(np.mean([p_culture_social, p_culture_majority]))
    p_age_avg = float(np.mean([p_age_social, p_age_majority]))

    total_score = 0
    total_score += contribution(p_culture_avg, delta_culture_avg)
    total_score += contribution(p_age_avg, delta_age_avg)

    # Clamp to Likert scale [-100, 100] and round to integer
    scalar = int(round(max(-100, min(100, total_score))))

    # Write final scalar conclusion
    Path("conclusion.txt").write_text(str(scalar), encoding="utf-8")

    # Print a brief summary for human inspection (not used for scoring)
    print(f"N observations: {len(df)}")
    print(f"Overall reliance on social info (follow_social): {df['follow_social'].mean():.3f}")
    print(f"Overall majority choice rate (choose_majority): {df['choose_majority'].mean():.3f}")
    print("p-values (age, culture) for social reliance:", p_age_social, p_culture_social)
    print("p-values (age, culture) for majority preference:", p_age_majority, p_culture_majority)
    print("Average effect sizes (culture, age):", delta_culture_avg, delta_age_avg)
    print("Final Likert scalar conclusion:", scalar)


if __name__ == "__main__":
    main()

