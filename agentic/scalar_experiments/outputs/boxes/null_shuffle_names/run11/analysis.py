import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcome coding:
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option.
    df = df.copy()
    df["is_majority"] = df["majority_first"] == 2
    df["is_minority"] = df["majority_first"] == 3
    df["is_undemonstrated"] = df["majority_first"] == 1
    # Reliance on social information = chose any demonstrated option (majority or minority).
    df["is_social"] = df["is_majority"] | df["is_minority"]

    # Helper to compute range of a rate across groups.
    def rate_range(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float(series.max() - series.min())

    # Developmental variation (across age in years).
    age_social_rates = df.groupby("age")["is_social"].mean()
    age_majority_rates = df.groupby("age")["is_majority"].mean()
    dev_social_range = rate_range(age_social_rates)
    dev_majority_range = rate_range(age_majority_rates)
    dev_variation = (dev_social_range + dev_majority_range) / 2.0

    # Cross-cultural variation (using site ID `y` as proxy for society).
    site_social_rates = df.groupby("y")["is_social"].mean()
    site_majority_rates = df.groupby("y")["is_majority"].mean()
    cult_social_range = rate_range(site_social_rates)
    cult_majority_range = rate_range(site_majority_rates)
    cult_variation = (cult_social_range + cult_majority_range) / 2.0

    # Overall variation index: how much majority use and social reliance vary
    # across both age and cultural site. This is bounded in [0, 1].
    variation_index = (dev_variation + cult_variation) / 2.0

    # Map variation index to a Likert-style scalar in [-100, 100].
    # - If variation is extremely small, treat this as strong evidence against
    #   meaningful differences (negative values).
    # - For moderate to large variation, map positively up to +100.
    if variation_index < 0.02:
        scalar = -100
    elif variation_index < 0.05:
        scalar = -60
    elif variation_index < 0.10:
        scalar = -20
    else:
        # For meaningful variation, scale linearly so that
        # variation_index = 0.10 -> 20, variation_index = 0.30 -> 60,
        # and variation_index >= 0.50 -> 100 (clipped).
        positive_score = min(variation_index, 0.50) / 0.50
        scalar = int(round(20 + positive_score * 80))

    # Ensure scalar is an int within [-100, 100].
    scalar = int(max(-100, min(100, scalar)))

    # Print a brief diagnostic summary for human inspection.
    print("Age social rate range:", dev_social_range)
    print("Age majority rate range:", dev_majority_range)
    print("Site social rate range:", cult_social_range)
    print("Site majority rate range:", cult_majority_range)
    print("Variation index:", variation_index)
    print("Likert scalar (final):", scalar)

    # Write scalar (and only the scalar) to conclusion.txt.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

