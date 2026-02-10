import numpy as np
import pandas as pd


def main() -> None:
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Outcome: 1 = undemonstrated option, 2 = majority, 3 = minority
    df["social"] = df["majority_first"] != 1
    df["majority_choice"] = df["majority_first"] == 2

    # Variation across cultural sites (y = site ID)
    site_social = df.groupby("y")["social"].mean()
    site_majority = df.groupby("y")["majority_choice"].mean()

    site_social_range = site_social.max() - site_social.min()
    site_majority_range = site_majority.max() - site_majority.min()

    # Variation across developmental stages (age groups)
    bins = [4, 7, 10, 13, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    age_social = df.groupby("age_group")["social"].mean()
    age_majority = df.groupby("age_group")["majority_choice"].mean()

    age_social_range = age_social.max() - age_social.min()
    age_majority_range = age_majority.max() - age_majority.min()

    # Aggregate variation index (0–1 scale, roughly)
    ranges = np.array(
        [site_social_range, site_majority_range, age_social_range, age_majority_range],
        dtype=float,
    )
    variation_index = float(np.nanmean(ranges))

    # Map variation index to Likert scalar [-100, 100], where positive = evidence for variation
    # A range of ~0.33 or more (33 percentage points) is treated as very strong evidence.
    scalar = int(round(max(0.0, min(variation_index, 0.3333)) * 300))

    # Ensure scalar is within bounds
    scalar = max(-100, min(100, scalar))

    # Write scalar conclusion
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))

    # Optional: print brief diagnostics for human inspection
    print("Site social ranges:", site_social_range, "Site majority ranges:", site_majority_range)
    print("Age social ranges:", age_social_range, "Age majority ranges:", age_majority_range)
    print("Variation index:", variation_index, "Scalar conclusion:", scalar)


if __name__ == "__main__":
    main()

