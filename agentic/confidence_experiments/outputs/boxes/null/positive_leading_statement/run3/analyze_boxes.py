import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Reliance on social information: 1 if child chose a demonstrated option (majority or minority), 0 otherwise.
    df["social_info"] = np.where(df["y"] == 1, 0, 1)

    # Preference for majority over minority, defined only for trials where a demonstrated option was chosen.
    df_social = df[df["social_info"] == 1].copy()
    df_social["majority_choice"] = np.where(df_social["y"] == 2, 1, 0)

    # Treat culture as categorical.
    # Model 1: Reliance on social information (full model).
    model_social = smf.logit(
        "social_info ~ age + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False)

    # Model 1a: Reliance on social information with culture only.
    model_social_culture_only = smf.logit(
        "social_info ~ C(culture)",
        data=df,
    ).fit(disp=False)

    # Model 2: Preference for majority over minority among children who used social info (full model).
    model_majority = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=df_social,
    ).fit(disp=False)

    # Model 2a: Preference for majority over minority with culture only.
    model_majority_culture_only = smf.logit(
        "majority_choice ~ C(culture)",
        data=df_social,
    ).fit(disp=False)

    # Basic descriptive statistics for interpretation.
    overall_social_rate = df["social_info"].mean()
    overall_majority_rate = df_social["majority_choice"].mean()

    # Age-grouped descriptives.
    age_bins = [4, 6, 8, 10, 12, 14]
    age_labels = ["4-5", "6-7", "8-9", "10-11", "12-13"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    df_social["age_group"] = pd.cut(
        df_social["age"], bins=age_bins, labels=age_labels, right=False
    )

    social_by_age = df.groupby("age_group")["social_info"].mean()
    majority_by_age = df_social.groupby("age_group")["majority_choice"].mean()

    social_by_culture = df.groupby("culture")["social_info"].mean()
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()

    # Print a concise textual report for manual inspection.
    print("=== Reliance on social information (any demonstrated option) ===")
    print(model_social.summary())
    print()
    print("Culture-only model for social-information use:")
    print(model_social_culture_only.summary())
    print()
    print(f"Overall social-information use rate: {overall_social_rate:.3f}")
    print("\nSocial-information use rate by age group:")
    print(social_by_age)
    print("\nSocial-information use rate by culture:")
    print(social_by_culture)

    print("\n=== Preference for majority over minority (conditional on social info use) ===")
    print(model_majority.summary())
    print()
    print("Culture-only model for majority preference:")
    print(model_majority_culture_only.summary())
    print()
    print(f"Overall majority-choice rate (among social-info users): {overall_majority_rate:.3f}")
    print("\nMajority-choice rate by age group:")
    print(majority_by_age)
    print("\nMajority-choice rate by culture:")
    print(majority_by_culture)


if __name__ == "__main__":
    main()
