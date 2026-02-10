import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic recodes
    df["is_social"] = df["y"].isin([2, 3])
    df["is_majority"] = df["y"] == 2
    df["is_minority"] = df["y"] == 3

    # Descriptive stats: social vs asocial by culture and age
    print("Overall choice distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    print("Social information use (choose demonstrated option 2 or 3) by culture:")
    social_by_culture = (
        df.groupby("culture")["is_social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_social"})
    )
    print(social_by_culture)
    print()

    print("Social information use by age (treated as integer years):")
    social_by_age = (
        df.groupby("age")["is_social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_social"})
    )
    print(social_by_age)
    print()

    # Majority vs minority among those who used social info
    social = df[df["is_social"]].copy()
    print("Among social choices only: majority vs minority proportions overall:")
    print(social["y"].value_counts(normalize=True).sort_index())
    print()

    print("Majority preference among social choosers, by culture:")
    maj_by_culture = (
        social.groupby("culture")["is_majority"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_majority"})
    )
    print(maj_by_culture)
    print()

    print("Majority preference among social choosers, by age:")
    maj_by_age = (
        social.groupby("age")["is_majority"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_majority"})
    )
    print(maj_by_age)
    print()

    # Simple age banding to make developmental trends easier to see
    bins = [4, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
    social["age_band"] = pd.cut(
        social["age"], bins=bins, labels=labels, include_lowest=True
    )

    print("Social information use by age band:")
    social_by_band = (
        df.groupby("age_band")["is_social"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_social"})
    )
    print(social_by_band)
    print()

    print("Majority preference among social choosers by age band:")
    maj_by_band = (
        social.groupby("age_band")["is_majority"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_majority"})
    )
    print(maj_by_band)
    print()

    # Rough measures of variability (range of proportions) across cultures and age bands
    social_culture_range = social_by_culture["prop_social"].max() - social_by_culture[
        "prop_social"
    ].min()
    social_band_range = social_by_band["prop_social"].max() - social_by_band[
        "prop_social"
    ].min()

    maj_culture_range = maj_by_culture["prop_majority"].max() - maj_by_culture[
        "prop_majority"
    ].min()
    maj_band_range = maj_by_band["prop_majority"].max() - maj_by_band[
        "prop_majority"
    ].min()

    print("Range of social-information use proportions across cultures:", social_culture_range)
    print("Range of social-information use proportions across age bands:", social_band_range)
    print("Range of majority-preference proportions across cultures:", maj_culture_range)
    print("Range of majority-preference proportions across age bands:", maj_band_range)


if __name__ == "__main__":
    main()

