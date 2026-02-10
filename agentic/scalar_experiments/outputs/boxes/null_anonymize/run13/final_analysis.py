from pathlib import Path

import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcomes
    # feature1: 1 = undemonstrated option, 2 = majority, 3 = minority
    df["social_reliance"] = df["feature1"].isin([2, 3])
    df["majority_choice"] = df["feature1"] == 2

    # Age bins (approximate developmental stages)
    age_bins = pd.cut(
        df["feature3"],
        bins=[4, 6, 8, 10, 12, 14],
        right=True,
        include_lowest=True,
    )

    # Social reliance variation across cultures (sites) and age bins
    social_by_site = df.groupby("feature5")["social_reliance"].mean()
    social_by_age = df.groupby(age_bins)["social_reliance"].mean()

    social_site_range = float(social_by_site.max() - social_by_site.min())
    social_age_range = float(social_by_age.max() - social_by_age.min())

    # Majority preference variation among children who use social information
    social_df = df[df["social_reliance"]]
    majority_by_site = social_df.groupby("feature5")["majority_choice"].mean()
    majority_by_age = social_df.groupby(age_bins)["majority_choice"].mean()

    majority_site_range = float(majority_by_site.max() - majority_by_site.min())
    majority_age_range = float(majority_by_age.max() - majority_by_age.min())

    # Average range as a summary measure of how much these tendencies vary
    avg_range = (
        social_site_range
        + social_age_range
        + majority_site_range
        + majority_age_range
    ) / 4.0

    # Map average proportion range in [0, 0.4+] into Likert [-100, 100],
    # but since this question is about whether variation exists, we keep
    # the sign positive and scale strength up to +80 when variation is large.
    capped_range = max(0.0, min(avg_range, 0.4))
    scalar = int(round((capped_range / 0.4) * 80))

    print("Social reliance range by site:", social_site_range)
    print("Social reliance range by age:", social_age_range)
    print("Majority preference range by site:", majority_site_range)
    print("Majority preference range by age:", majority_age_range)
    print("Average range:", avg_range)
    print("Scalar conclusion (-100 to 100):", scalar)

    # Write scalar to conclusion.txt as required (single integer only)
    Path("conclusion.txt").write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

