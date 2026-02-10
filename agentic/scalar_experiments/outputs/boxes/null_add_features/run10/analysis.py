import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode outcome categories
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df = df[df["y"].isin([1, 2, 3])].copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)

    # Basic overall tendencies
    overall_majority_rate = df["majority_choice"].mean()
    overall_social_rate = df["social_choice"].mean()

    # Variation across cultures
    culture_rates = (
        df.groupby("culture")[["majority_choice", "social_choice"]].mean()
    )
    majority_range_culture = (
        culture_rates["majority_choice"].max()
        - culture_rates["majority_choice"].min()
    )
    social_range_culture = (
        culture_rates["social_choice"].max()
        - culture_rates["social_choice"].min()
    )

    # Age effects: treat age as continuous; inspect correlation with majority_choice
    if df["age"].notna().any():
        age_majority_corr = df[["age", "majority_choice"]].corr().iloc[0, 1]
        age_social_corr = df[["age", "social_choice"]].corr().iloc[0, 1]
    else:
        age_majority_corr = 0.0
        age_social_corr = 0.0

    # Summarise effect magnitudes into a heuristic "evidence for variation" score
    # Stronger between-culture ranges and age correlations -> stronger evidence
    # Weight majority preference more heavily than generic social choice.
    culture_component = (
        0.6 * majority_range_culture + 0.4 * social_range_culture
    )
    age_component = 0.6 * abs(age_majority_corr) + 0.4 * abs(age_social_corr)

    # Combine components, scaled into [0, 1]
    # Typical behavioral ranges: majority_range up to ~0.7, correlations up to ~0.5
    norm_culture = min(culture_component / 0.7, 1.0)
    norm_age = min(age_component / 0.5, 1.0)

    # Evidence score is the mean of culture and age components
    evidence_score = (norm_culture + norm_age) / 2.0

    # Map evidence_score in [0,1] to Likert [-100, 100],
    # where higher means stronger "Yes, reliance and preference vary".
    scalar = int(round(evidence_score * 200 - 100))

    # Clip to ensure within [-100, 100]
    scalar = int(np.clip(scalar, -100, 100))

    # Write scalar to conclusion.txt as required (single integer, no extra text)
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

