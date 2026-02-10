import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Define broad developmental stages
    bins = [4, 7, 10, 13, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    print("Overall outcome distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    # Reliance on social information (any demonstrated option) by culture and age group
    print("Reliance on social information by culture:")
    social_by_culture = (
        df.groupby("culture")["social_choice"].mean().rename("p_social")
    )
    print(social_by_culture)
    print()

    print("Reliance on social information by age group:")
    social_by_age = df.groupby("age_group")["social_choice"].mean().rename("p_social")
    print(social_by_age)
    print()

    # Preference for majority vs minority among social choices
    social_df = df[df["social_choice"] == 1].copy()

    print("Majority preference among social choices by culture:")
    maj_by_culture = (
        social_df.groupby("culture")["majority_choice"].mean().rename("p_majority")
    )
    print(maj_by_culture)
    print()

    print("Majority preference among social choices by age group:")
    maj_by_age = (
        social_df.groupby("age_group")["majority_choice"].mean().rename("p_majority")
    )
    print(maj_by_age)
    print()

    # Simple ranges to quantify heterogeneity
    print("Range of social reliance across cultures:", social_by_culture.max() - social_by_culture.min())
    print("Range of social reliance across age groups:", social_by_age.max() - social_by_age.min())
    print("Range of majority preference across cultures:", maj_by_culture.max() - maj_by_culture.min())
    print("Range of majority preference across age groups:", maj_by_age.max() - maj_by_age.min())


if __name__ == "__main__":
    main()

