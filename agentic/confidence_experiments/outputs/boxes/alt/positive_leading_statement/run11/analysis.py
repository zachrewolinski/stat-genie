import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full_model, reduced_model):
    """Likelihood ratio test comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    return float(lr_stat), int(df_diff), float(p_value)


def main():
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social_use"] = (df["y"] != 1).astype(int)
    df["majority_choice_any"] = (df["y"] == 2).astype(int)

    social_mask = df["y"] != 1
    df_social = df.loc[social_mask].copy()
    df_social["majority_choice_social"] = (df_social["y"] == 2).astype(int)

    # Age groups for descriptive summaries (developmental stages)
    bins = [4, 7, 10, 13, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df_social["age_group"] = pd.cut(df_social["age"], bins=bins, labels=labels, right=False)

    # GLM (binomial) models for reliance on social information
    social_full = smf.glm(
        "social_use ~ age + C(culture)", data=df, family=sm.families.Binomial()
    ).fit()
    social_age_only = smf.glm(
        "social_use ~ age", data=df, family=sm.families.Binomial()
    ).fit()

    social_age_p = float(social_full.pvalues["age"])
    social_lr_stat, social_df_diff, social_culture_p = lr_test(
        social_full, social_age_only
    )

    # Effect of age on social_use: predicted probability from youngest to oldest
    age_min = df["age"].min()
    age_max = df["age"].max()
    df_low = df.copy()
    df_low["age"] = age_min
    df_high = df.copy()
    df_high["age"] = age_max
    social_prob_low = float(social_full.predict(df_low).mean())
    social_prob_high = float(social_full.predict(df_high).mean())
    social_age_diff = social_prob_high - social_prob_low

    # Cultural range in reliance on social information (empirical proportions)
    social_by_culture = df.groupby("culture")["social_use"].mean()
    social_culture_min = float(social_by_culture.min())
    social_culture_max = float(social_by_culture.max())
    social_culture_range = social_culture_max - social_culture_min

    # Models for majority preference among all choices
    maj_any_full = smf.glm(
        "majority_choice_any ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    maj_any_age_only = smf.glm(
        "majority_choice_any ~ age", data=df, family=sm.families.Binomial()
    ).fit()

    maj_any_age_p = float(maj_any_full.pvalues["age"])
    maj_any_lr_stat, maj_any_df_diff, maj_any_culture_p = lr_test(
        maj_any_full, maj_any_age_only
    )

    df_low2 = df.copy()
    df_low2["age"] = age_min
    df_high2 = df.copy()
    df_high2["age"] = age_max
    maj_any_prob_low = float(maj_any_full.predict(df_low2).mean())
    maj_any_prob_high = float(maj_any_full.predict(df_high2).mean())
    maj_any_age_diff = maj_any_prob_high - maj_any_prob_low

    maj_any_by_culture = df.groupby("culture")["majority_choice_any"].mean()
    maj_any_culture_min = float(maj_any_by_culture.min())
    maj_any_culture_max = float(maj_any_by_culture.max())
    maj_any_culture_range = maj_any_culture_max - maj_any_culture_min

    # Models for majority preference conditional on using social information
    maj_social_full = smf.glm(
        "majority_choice_social ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    maj_social_age_only = smf.glm(
        "majority_choice_social ~ age", data=df_social, family=sm.families.Binomial()
    ).fit()

    maj_social_age_p = float(maj_social_full.pvalues["age"])
    maj_social_lr_stat, maj_social_df_diff, maj_social_culture_p = lr_test(
        maj_social_full, maj_social_age_only
    )

    df_social_low = df_social.copy()
    df_social_low["age"] = age_min
    df_social_high = df_social.copy()
    df_social_high["age"] = age_max
    maj_social_prob_low = float(maj_social_full.predict(df_social_low).mean())
    maj_social_prob_high = float(maj_social_full.predict(df_social_high).mean())
    maj_social_age_diff = maj_social_prob_high - maj_social_prob_low

    maj_social_by_culture = df_social.groupby("culture")["majority_choice_social"].mean()
    maj_social_culture_min = float(maj_social_by_culture.min())
    maj_social_culture_max = float(maj_social_by_culture.max())
    maj_social_culture_range = maj_social_culture_max - maj_social_culture_min

    # Age-grouped descriptive statistics
    social_by_age_group = df.groupby("age_group")["social_use"].mean().to_dict()
    maj_social_by_age_group = (
        df_social.groupby("age_group")["majority_choice_social"].mean().to_dict()
    )

    # Collect key metrics for inspection
    results = {
        "n": int(len(df)),
        "age_range": [float(age_min), float(age_max)],
        "social_use": {
            "age_p": social_age_p,
            "culture_lr_p": social_culture_p,
            "prob_young": social_prob_low,
            "prob_old": social_prob_high,
            "age_diff": social_age_diff,
            "culture_min": social_culture_min,
            "culture_max": social_culture_max,
            "culture_range": social_culture_range,
            "by_age_group": {str(k): float(v) for k, v in social_by_age_group.items()},
        },
        "majority_any": {
            "age_p": maj_any_age_p,
            "culture_lr_p": maj_any_culture_p,
            "prob_young": maj_any_prob_low,
            "prob_old": maj_any_prob_high,
            "age_diff": maj_any_age_diff,
            "culture_min": maj_any_culture_min,
            "culture_max": maj_any_culture_max,
            "culture_range": maj_any_culture_range,
        },
        "majority_social": {
            "age_p": maj_social_age_p,
            "culture_lr_p": maj_social_culture_p,
            "prob_young": maj_social_prob_low,
            "prob_old": maj_social_prob_high,
            "age_diff": maj_social_age_diff,
            "culture_min": maj_social_culture_min,
            "culture_max": maj_social_culture_max,
            "culture_range": maj_social_culture_range,
            "by_age_group": {
                str(k): float(v) for k, v in maj_social_by_age_group.items()
            },
        },
    }

    # Print results in JSON for easy inspection
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

