import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcome into social-information reliance and majority preference
    # y: 1=unchosen (asocial), 2=majority, 3=minority
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Overall reliance on social info and majority preference
    social_rate = df["social"].mean()
    majority_rate = df["majority_choice"].mean()

    # Age trends (correlation with age)
    age = df["age"]
    social_age_corr = np.corrcoef(age, df["social"])[0, 1]
    majority_age_corr = np.corrcoef(age, df["majority_choice"])[0, 1]

    # Culture-level majority preference
    culture_majority = df.groupby("culture")["majority_choice"].mean()
    culture_social = df.groupby("culture")["social"].mean()

    # Simple quantitative summary to support a scalar Likert judgment:
    # - strong evidence for reliance on social information if social_rate >> 0.5
    # - strong majority preference if majority_rate >> 0.5 across cultures/ages

    # Map findings to a Likert-style scalar in [-100, 100].
    # We start from 0 (neutral) and add evidence-based increments.
    score = 0

    # Overall social reliance
    if social_rate > 0.8:
        score += 40
    elif social_rate > 0.65:
        score += 25
    elif social_rate > 0.55:
        score += 10
    elif social_rate < 0.45:
        score -= 20

    # Overall majority preference strength
    if majority_rate > 0.7:
        score += 35
    elif majority_rate > 0.6:
        score += 20
    elif majority_rate > 0.5:
        score += 10
    elif majority_rate < 0.4:
        score -= 20

    # Age-related modulation of social/majority reliance
    # Positive correlations support developmental variation in majority use.
    if social_age_corr > 0.15:
        score += 10
    elif social_age_corr < -0.15:
        score -= 10

    if majority_age_corr > 0.15:
        score += 10
    elif majority_age_corr < -0.15:
        score -= 10

    # Cross-cultural variation: if cultures differ meaningfully, this supports
    # the "vary across cultures" component of the question.
    if culture_majority.std() > 0.1 or culture_social.std() > 0.1:
        score += 10

    # Clip to [-100, 100] and round to nearest integer
    score = int(np.clip(round(score), -100, 100))

    with open("analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"social_rate={social_rate:.3f}\n"
            f"majority_rate={majority_rate:.3f}\n"
            f"social_age_corr={social_age_corr:.3f}\n"
            f"majority_age_corr={majority_age_corr:.3f}\n"
            f"culture_majority_mean={culture_majority.mean():.3f}\n"
            f"culture_majority_std={culture_majority.std():.3f}\n"
            f"culture_social_mean={culture_social.mean():.3f}\n"
            f"culture_social_std={culture_social.std():.3f}\n"
            f"computed_score={score}\n"
        )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()

