import pandas as pd
import numpy as np


def main() -> None:
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Outcome recodes
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = (df["majority_first"] != 1).astype(int)

    # Define age groups for developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    # Helper to compute range of proportions by grouping variable
    def prop_range(data: pd.DataFrame, outcome: str, group: str) -> float:
        grouped = data.groupby(group)[outcome].mean()
        if grouped.empty:
            return 0.0
        return float(grouped.max() - grouped.min())

    # Variation in reliance on social information (social_choice)
    social_site_range = prop_range(df, "social_choice", "y")
    social_age_range = prop_range(df, "social_choice", "age_group")

    # Variation in preference for majority cues (majority_choice)
    majority_site_range = prop_range(df, "majority_choice", "y")
    majority_age_range = prop_range(df, "majority_choice", "age_group")

    # Aggregate variation metric in [0, 1]
    variation_components = [
        social_site_range,
        social_age_range,
        majority_site_range,
        majority_age_range,
    ]
    variation_score = float(np.mean(variation_components))
    variation_score = max(0.0, min(1.0, variation_score))

    # Determine sign for Likert scale:
    # - If variation is very small, treat as (weak) evidence against variation.
    # - Otherwise, treat as evidence in favor of variation.
    if variation_score < 0.05:
        sign = -1
    elif variation_score > 0.1:
        sign = 1
    else:
        sign = 0

    likert_value = int(round(variation_score * 100 * sign))
    likert_value = max(-100, min(100, likert_value))

    # Write scalar conclusion to file with no extra text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(likert_value))


if __name__ == "__main__":
    main()

