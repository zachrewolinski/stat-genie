import pandas as pd
import numpy as np


def main() -> None:
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Encode outcomes
    # majority_first: 1 = undemonstrated (asocial), 2 = majority, 3 = minority
    df["social_use"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = df["majority_first"].map({2: 1, 3: 0})

    # Treat site/culture identifier as categorical
    df["site"] = df["y"].astype("category")

    # Basic overall descriptives
    n = len(df)
    overall_social = df["social_use"].mean()
    majority_only = df.dropna(subset=["majority_choice"])
    overall_majority_pref = majority_only["majority_choice"].mean()

    print("N children:", n)
    print(f"Overall reliance on social information (any demonstrator): {overall_social:.3f}")
    print(f"Overall majority preference among social choices: {overall_majority_pref:.3f}")
    print()

    # Variation across developmental stages (age in years)
    age_social_means = df.groupby("age")["social_use"].mean()
    age_majority_means = majority_only.groupby("age")["majority_choice"].mean()

    print("Reliance on social information by age (mean probability):")
    print(age_social_means)
    print()

    print("Majority preference by age (conditional on using social information):")
    print(age_majority_means)
    print()

    # Variation across cultures/sites
    site_social_means = df.groupby("site")["social_use"].mean()
    site_majority_means = majority_only.groupby("site")["majority_choice"].mean()

    print("Reliance on social information by site:")
    print(site_social_means)
    print()

    print("Majority preference by site:")
    print(site_majority_means)
    print()

    # Quantify magnitude of variation (range of probabilities)
    def safe_range(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float(series.max() - series.min())

    range_age_social = safe_range(age_social_means)
    range_age_majority = safe_range(age_majority_means)
    range_site_social = safe_range(site_social_means)
    range_site_majority = safe_range(site_majority_means)

    print("Range of probabilities (max - min) across ages and sites:")
    print(f"  Social use by age:      {range_age_social:.3f}")
    print(f"  Majority pref by age:   {range_age_majority:.3f}")
    print(f"  Social use by site:     {range_site_social:.3f}")
    print(f"  Majority pref by site:  {range_site_majority:.3f}")
    print()

    # Aggregate variation metric (0–1)
    variation_components = [
        range_age_social,
        range_age_majority,
        range_site_social,
        range_site_majority,
    ]
    variation_score = float(np.mean(variation_components))

    print(f"Average variation score (0–1 scale): {variation_score:.3f}")


if __name__ == "__main__":
    main()

