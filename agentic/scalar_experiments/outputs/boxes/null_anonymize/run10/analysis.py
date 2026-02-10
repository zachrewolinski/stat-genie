import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Basic overall proportions
    choice_counts = df["choice"].value_counts(normalize=True).sort_index()

    # Social information use: any demonstrated option (majority or minority)
    social_use = df["choice"].isin([2, 3]).mean()
    majority_pref_overall = (df["choice"] == 2).mean()

    # Age groups to examine developmental trends
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    age_group_summary = (
        df.groupby("age_group")["choice"]
        .value_counts(normalize=True)
        .rename("prop")
        .reset_index()
        .pivot(index="age_group", columns="choice", values="prop")
    )

    # Site-level majority preference
    site_majority_pref = df.groupby("site")["choice"].apply(
        lambda s: (s == 2).mean()
    )

    # Simple effect-size style summaries for variation
    age_majority_range = (
        age_group_summary.get(2, pd.Series(dtype=float)).max()
        - age_group_summary.get(2, pd.Series(dtype=float)).min()
    )
    site_majority_range = site_majority_pref.max() - site_majority_pref.min()

    print("Overall choice proportions (1=undemonstrated, 2=majority, 3=minority):")
    print(choice_counts)
    print()
    print(f"Overall social information use (2 or 3): {social_use:.3f}")
    print(f"Overall majority preference (choice==2): {majority_pref_overall:.3f}")
    print()
    print("Age group majority/minority/undemonstrated proportions:")
    print(age_group_summary)
    print()
    print("Site-level majority preference:")
    print(site_majority_pref)
    print()
    print(f"Range in majority preference across age groups: {age_majority_range:.3f}")
    print(f"Range in majority preference across sites: {site_majority_range:.3f}")


if __name__ == "__main__":
    main()

