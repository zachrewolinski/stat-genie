import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define derived variables
    df["social_choice"] = np.where(df["y"].isin([2, 3]), 1, 0)
    df_social = df[df["social_choice"] == 1].copy()
    df_social["majority_choice"] = np.where(df_social["y"] == 2, 1, 0)

    # Age groups for developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )
    df_social["age_group"] = pd.cut(
        df_social["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    # 1) Reliance on social information (any demonstrated option vs undemonstrated)
    ct_culture_social = pd.crosstab(df["culture"], df["social_choice"])
    chi2_culture_social, p_culture_social, dof1, exp1 = stats.chi2_contingency(
        ct_culture_social
    )

    ct_age_social = pd.crosstab(df["age_group"], df["social_choice"])
    chi2_age_social, p_age_social, dof2, exp2 = stats.chi2_contingency(ct_age_social)

    # 2) Preference for majority vs minority (among social choices only)
    ct_culture_majority = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    chi2_culture_majority, p_culture_majority, dof3, exp3 = stats.chi2_contingency(
        ct_culture_majority
    )

    ct_age_majority = pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    chi2_age_majority, p_age_majority, dof4, exp4 = stats.chi2_contingency(
        ct_age_majority
    )

    # Simple descriptive summaries
    social_by_culture = (
        df.groupby("culture")["social_choice"].mean().rename("social_rate")
    )
    social_by_age = df.groupby("age_group")["social_choice"].mean().rename("social_rate")

    majority_by_culture = (
        df_social.groupby("culture")["majority_choice"].mean().rename("majority_rate")
    )
    majority_by_age = (
        df_social.groupby("age_group")["majority_choice"]
        .mean()
        .rename("majority_rate")
    )

    # Print a concise summary for inspection
    print("Reliance on social information (social_choice=1):")
    print("  Culture vs social_choice chi2, p:", chi2_culture_social, p_culture_social)
    print(ct_culture_social)
    print("\n  Social-choice rate by culture:")
    print(social_by_culture)

    print("\n  Age-group vs social_choice chi2, p:", chi2_age_social, p_age_social)
    print(ct_age_social)
    print("\n  Social-choice rate by age_group:")
    print(social_by_age)

    print("\nPreference for majority (majority_choice=1 among social choices):")
    print(
        "  Culture vs majority_choice chi2, p:",
        chi2_culture_majority,
        p_culture_majority,
    )
    print(ct_culture_majority)
    print("\n  Majority-choice rate by culture:")
    print(majority_by_culture)

    print(
        "\n  Age-group vs majority_choice chi2, p:",
        chi2_age_majority,
        p_age_majority,
    )
    print(ct_age_majority)
    print("\n  Majority-choice rate by age_group:")
    print(majority_by_age)

    # Also dump a small JSON summary to inspect programmatically if needed.
    summary = {
        "p_values": {
            "culture_social": float(p_culture_social),
            "age_social": float(p_age_social),
            "culture_majority": float(p_culture_majority),
            "age_majority": float(p_age_majority),
        },
        "descriptives": {
            "social_rate_by_culture": social_by_culture.to_dict(),
            "social_rate_by_age_group": social_by_age.to_dict(),
            "majority_rate_by_culture": majority_by_culture.to_dict(),
            "majority_rate_by_age_group": majority_by_age.to_dict(),
        },
    }
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

