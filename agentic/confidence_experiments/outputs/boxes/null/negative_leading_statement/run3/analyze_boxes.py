import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(full_result, reduced_result):
    """Likelihood ratio test between nested models."""
    lr_stat = 2.0 * (full_result.llf - reduced_result.llf)
    df_diff = full_result.df_model - reduced_result.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, int(df_diff), float(p_value)


def main():
    df = pd.read_csv("boxes.csv")

    # Reliance on social information: choosing any demonstrated option (majority or minority)
    df["social"] = (df["y"] != 1).astype(int)

    # Preference for majority cues: among social choices, majority (2) vs minority (3)
    df["majority_choice"] = df["y"].map({2: 1, 3: 0})

    # Basic descriptive summaries
    overall_social_rate = df["social"].mean()
    social_by_culture = df.groupby("culture")["social"].mean()

    social_df = df[df["majority_choice"].notna()].copy()
    overall_majority_rate = social_df["majority_choice"].mean()
    majority_by_culture = social_df.groupby("culture")["majority_choice"].mean()

    # Age groups for descriptive summaries
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    age_social = df.groupby("age_group")["social"].mean()
    age_majority = (
        df[df["majority_choice"].notna()]
        .groupby("age_group")["majority_choice"]
        .mean()
    )

    # Logistic models for social reliance
    social_full = smf.logit(
        "social ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False, maxiter=1000)
    social_no_culture = smf.logit(
        "social ~ age + gender + majority_first", data=df
    ).fit(disp=False, maxiter=1000)
    social_no_age = smf.logit(
        "social ~ C(culture) + gender + majority_first", data=df
    ).fit(disp=False, maxiter=1000)

    lr_social_culture = lr_test(social_full, social_no_culture)
    lr_social_age = lr_test(social_full, social_no_age)
    social_age_coef = social_full.params.get("age", float("nan"))

    # Interaction of age with culture for social reliance
    try:
        social_interact = smf.logit(
            "social ~ age * C(culture) + gender + majority_first", data=df
        ).fit(disp=False, maxiter=1000)
        lr_social_interact = lr_test(social_interact, social_full)
    except Exception:
        social_interact = None
        lr_social_interact = None

    # Logistic models for majority preference (only among social choices)
    maj_full = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first", data=social_df
    ).fit(disp=False, maxiter=1000)
    maj_no_culture = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=social_df
    ).fit(disp=False, maxiter=1000)
    maj_no_age = smf.logit(
        "majority_choice ~ C(culture) + gender + majority_first", data=social_df
    ).fit(disp=False, maxiter=1000)

    lr_maj_culture = lr_test(maj_full, maj_no_culture)
    lr_maj_age = lr_test(maj_full, maj_no_age)
    maj_age_coef = maj_full.params.get("age", float("nan"))

    # Interaction of age with culture for majority preference
    try:
        maj_interact = smf.logit(
            "majority_choice ~ age * C(culture) + gender + majority_first",
            data=social_df,
        ).fit(disp=False, maxiter=1000)
        lr_maj_interact = lr_test(maj_interact, maj_full)
    except Exception:
        maj_interact = None
        lr_maj_interact = None

    # Print a compact summary that we can inspect from the shell
    print("Overall social reliance rate:", round(overall_social_rate, 3))
    print("Social reliance by culture:", social_by_culture.round(3).to_dict())
    print("Social reliance by age group:", age_social.round(3).to_dict())
    print("LR test social ~ culture (chi2, df, p):", lr_social_culture)
    print("LR test social ~ age (chi2, df, p):", lr_social_age)
    print("Logit coef age -> social:", social_age_coef)
    print("LR test social age*culture interaction (chi2, df, p):", lr_social_interact)
    print()
    print("Overall majority preference rate (among social choices):", round(overall_majority_rate, 3))
    print("Majority preference by culture:", majority_by_culture.round(3).to_dict())
    print("Majority preference by age group:", age_majority.round(3).to_dict())
    print("LR test majority ~ culture (chi2, df, p):", lr_maj_culture)
    print("LR test majority ~ age (chi2, df, p):", lr_maj_age)
    print("Logit coef age -> majority:", maj_age_coef)
    print("LR test majority age*culture interaction (chi2, df, p):", lr_maj_interact)


if __name__ == "__main__":
    main()
