import pandas as pd
import numpy as np


def main() -> None:
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Basic cleaning: drop rows missing key fields
    df = df.dropna(subset=["majority_first", "age", "y"])

    # Encode key behavioral measures
    df["social_choice"] = (df["majority_first"] != 1).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["majority_given_social"] = np.where(
        df["social_choice"] == 1, df["majority_choice"], np.nan
    )

    # Overall summaries (printed for transparency/debugging)
    overall_social = df["social_choice"].mean()
    overall_majority = df["majority_choice"].mean()

    print(f"Overall social-information use rate: {overall_social:.3f}")
    print(f"Overall majority-choice rate: {overall_majority:.3f}")

    # Variation across ages (developmental stage)
    social_by_age = df.groupby("age")["social_choice"].mean()
    majority_by_age = df.groupby("age")["majority_given_social"].mean()

    # Variation across sites (cultural context)
    social_by_site = df.groupby("y")["social_choice"].mean()
    majority_by_site = df.groupby("y")["majority_given_social"].mean()

    variations = []

    if len(social_by_age) > 1:
        var_social_age = social_by_age.max() - social_by_age.min()
        print(f"Variation in social use across ages: {var_social_age:.3f}")
        variations.append(var_social_age)

    if len(majority_by_age.dropna()) > 1:
        var_majority_age = majority_by_age.max() - majority_by_age.min()
        print(f"Variation in majority bias across ages: {var_majority_age:.3f}")
        variations.append(var_majority_age)

    if len(social_by_site) > 1:
        var_social_site = social_by_site.max() - social_by_site.min()
        print(f"Variation in social use across sites: {var_social_site:.3f}")
        variations.append(var_social_site)

    if len(majority_by_site.dropna()) > 1:
        var_majority_site = majority_by_site.max() - majority_by_site.min()
        print(f"Variation in majority bias across sites: {var_majority_site:.3f}")
        variations.append(var_majority_site)

    if variations:
        # Average magnitude of variation across all relevant metrics
        avg_variation = float(np.mean(variations))
    else:
        avg_variation = 0.0

    print(f"Average variation magnitude: {avg_variation:.3f}")

    # Map empirical variation (0–~1) to Likert-style [-100, 100]
    # Use 0.30 as a "strong" variation benchmark.
    intensity = avg_variation / 0.30
    intensity = max(0.0, min(1.0, intensity))

    scalar = int(round(100 * intensity))

    # Positive values indicate evidence that reliance on social information
    # and majority preference DO vary across cultures and developmental stages.
    # Negative values would have indicated strong evidence of invariance,
    # but with the variation-based mapping, we stay on the [0, 100] side.

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))

    print(f"Scalar conclusion written to conclusion.txt: {scalar}")


if __name__ == "__main__":
    main()

