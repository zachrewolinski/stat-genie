import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    info_path = Path("info.json")

    df = pd.read_csv(data_path)

    # Basic sanity checks
    print(f"Loaded data with shape: {df.shape}")
    print("Head:")
    print(df.head())
    print()

    # Create derived variables
    df["social"] = (df["y"] != 1).astype(int)
    followers = df[df["social"] == 1].copy()
    followers["majority_choice"] = (followers["y"] == 2).astype(int)

    print(f"Total children: {len(df)}")
    print(
        f"Overall proportion using social information (majority or minority): "
        f"{df['social'].mean():.3f}"
    )
    print(
        "Among social learners, proportion choosing majority option: "
        f"{followers['majority_choice'].mean():.3f}"
    )
    print()

    # Logistic regression for reliance on social information
    print("=== Logistic regression: social information use (social vs non-social) ===")
    social_full = smf.logit(
        "social ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False)
    social_no_culture = smf.logit(
        "social ~ age + gender + majority_first", data=df
    ).fit(disp=False)

    print(social_full.summary())

    # Age effect from Wald test (coefficient p-value)
    age_p_social = social_full.pvalues.get("age", np.nan)

    # Culture effect from likelihood-ratio test
    lr_stat_social = 2 * (social_full.llf - social_no_culture.llf)
    df_diff_social = social_full.df_model - social_no_culture.df_model
    p_culture_social = stats.chi2.sf(lr_stat_social, df_diff_social)

    print(
        f"\nSocial info use: age coefficient p-value = {age_p_social:.4g}, "
        f"culture LR test: chi2({df_diff_social:.0f}) = {lr_stat_social:.3f}, "
        f"p = {p_culture_social:.4g}"
    )
    print()

    # Logistic regression for majority vs minority choice among social learners
    print("=== Logistic regression: majority vs minority choice (among social learners) ===")
    majority_full = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first", data=followers
    ).fit(disp=False)
    majority_no_culture = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=followers
    ).fit(disp=False)

    print(majority_full.summary())

    age_p_majority = majority_full.pvalues.get("age", np.nan)
    lr_stat_majority = 2 * (majority_full.llf - majority_no_culture.llf)
    df_diff_majority = majority_full.df_model - majority_no_culture.df_model
    p_culture_majority = stats.chi2.sf(lr_stat_majority, df_diff_majority)

    print(
        f"\nMajority preference: age coefficient p-value = {age_p_majority:.4g}, "
        f"culture LR test: chi2({df_diff_majority:.0f}) = {lr_stat_majority:.3f}, "
        f"p = {p_culture_majority:.4g}"
    )
    print()

    # Also provide simple descriptive stats by age and culture for context
    df["age_group"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    followers["age_group"] = pd.cut(
        followers["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"]
    )

    print("Proportion using social information by age group:")
    print(df.groupby("age_group")["social"].mean())
    print("\nProportion choosing majority (among social learners) by age group:")
    print(followers.groupby("age_group")["majority_choice"].mean())
    print("\nProportion using social information by culture:")
    print(df.groupby("culture")["social"].mean())
    print("\nProportion choosing majority (among social learners) by culture:")
    print(followers.groupby("culture")["majority_choice"].mean())


if __name__ == "__main__":
    main()

