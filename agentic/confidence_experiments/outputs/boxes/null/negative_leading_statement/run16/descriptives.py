from pathlib import Path

import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")
    df["social"] = (df["y"] != 1).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Proportion relying on social information by culture
    social_by_culture = df.groupby("culture")["social"].mean()
    # Proportion choosing majority (among social choices) by culture
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()

    # Proportion relying on social information by age (binned)
    df["age_bin"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    social_by_agebin = df.groupby("age_bin")["social"].mean()

    df_social["age_bin"] = pd.cut(df_social["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    majority_by_agebin = df_social.groupby("age_bin")["majority_choice"].mean()

    print("Social reliance by culture:")
    print(social_by_culture.to_string())
    print("\nMajority choice (among social) by culture:")
    print(majority_by_culture.to_string())
    print("\nSocial reliance by age bin:")
    print(social_by_agebin.to_string())
    print("\nMajority choice (among social) by age bin:")
    print(majority_by_agebin.to_string())


if __name__ == "__main__":
    main()

