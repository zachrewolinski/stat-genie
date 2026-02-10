import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic sanity checks
    print("N rows:", len(df))
    print("Columns:", list(df.columns))

    # Outcome encoding
    # y: 1=undemonstrated option, 2=majority option, 3=minority option
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Overall outcome distribution
    print("\nOverall outcome distribution (y):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nOverall reliance on social information (social_choice):")
    print(df["social_choice"].mean())

    print("\nOverall preference for majority among all trials (majority_choice):")
    print(df["majority_choice"].mean())

    # Restrict to trials where a social option was chosen to examine majority vs minority
    social_mask = df["y"].isin([2, 3])
    df_social = df.loc[social_mask].copy()
    if len(df_social) > 0:
        print("\nAmong social choices only:")
        print("Proportion majority (y=2):", (df_social["y"] == 2).mean())
        print("Proportion minority (y=3):", (df_social["y"] == 3).mean())

    # By culture
    if "culture" in df.columns:
        print("\nReliance on social information by culture (mean social_choice):")
        social_by_culture = df.groupby("culture")["social_choice"].mean()
        print(social_by_culture)
        print("Std across cultures:", social_by_culture.std())

        print("\nMajority preference by culture (mean majority_choice):")
        majority_by_culture = df.groupby("culture")["majority_choice"].mean()
        print(majority_by_culture)
        print("Std across cultures:", majority_by_culture.std())

    # By age (treated as continuous)
    if "age" in df.columns:
        print("\nCorrelation of age with social_choice:")
        corr_age_social = df[["age", "social_choice"]].corr().iloc[0, 1]
        print(corr_age_social)

        print("\nCorrelation of age with majority_choice:")
        corr_age_majority = df[["age", "majority_choice"]].corr().iloc[0, 1]
        print(corr_age_majority)

        # Age groups (quartiles)
        df["age_group"] = pd.qcut(df["age"], q=4, duplicates="drop")
        print("\nReliance on social information by age_group:")
        print(df.groupby("age_group")["social_choice"].mean())

        print("\nMajority preference by age_group:")
        print(df.groupby("age_group")["majority_choice"].mean())


if __name__ == "__main__":
    main()

