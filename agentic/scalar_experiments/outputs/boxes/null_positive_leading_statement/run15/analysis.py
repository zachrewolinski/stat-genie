import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcome into indicators
    df["is_majority"] = (df["y"] == 2).astype(int)
    df["is_minority"] = (df["y"] == 3).astype(int)
    df["is_other"] = (df["y"] == 1).astype(int)

    overall_majority_rate = df["is_majority"].mean()

    # Majority preference by culture
    culture_rates = (
        df.groupby("culture")["is_majority"]
        .mean()
        .rename("majority_rate")
        .to_frame()
    )

    # Majority preference by age (treat age as integer years)
    age_rates = (
        df.groupby("age")["is_majority"]
        .mean()
        .rename("majority_rate")
        .to_frame()
    )

    # Simple measures of variability across cultures and ages
    culture_std = culture_rates["majority_rate"].std(ddof=0)
    age_std = age_rates["majority_rate"].std(ddof=0)

    # Correlation between age and majority choice at the individual level
    age_majority_corr = df[["age", "is_majority"]].corr().iloc[0, 1]

    # Heuristic scores for cultural and developmental variation
    # Scale variability (0-0.5 range roughly) into 0-1
    culture_variation_score = min(culture_std / 0.15, 1.0)
    age_variation_score = min(abs(age_majority_corr) / 0.3, 1.0)

    # Base endorsement reflects strong overall majority preference
    # Map majority rate (0.33-1) into (0-1)
    base_endorsement = max((overall_majority_rate - 1 / 3) / (1 - 1 / 3), 0.0)

    # Combine evidence: strong weight on overall majority following,
    # plus contributions from culture and age variation.
    combined_score = (
        0.5 * base_endorsement
        + 0.25 * culture_variation_score
        + 0.25 * age_variation_score
    )

    scalar = int(round(combined_score * 100))

    # Ensure scalar is within [-100, 100]; here it should be non-negative.
    scalar = max(min(scalar, 100), -100)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

