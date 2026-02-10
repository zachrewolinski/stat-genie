import numpy as np
import pandas as pd


def main() -> None:
    # Load dataset
    df = pd.read_csv("boxes.csv")

    # Encode key behavioral outcomes
    # social_use: 1 if child followed any demonstrated option (majority or minority)
    df["social_use"] = df["feature1"].isin([2, 3]).astype(int)
    # majority_choice: 1 if child followed the majority option
    df["majority_choice"] = (df["feature1"] == 2).astype(int)

    # ----- Variation across cultures (sites) -----
    site_groups = df.groupby("feature5")
    site_social = site_groups["social_use"].mean()

    social_users = df[df["social_use"] == 1]
    site_majority = social_users.groupby("feature5")["majority_choice"].mean()

    v_site_social = float(site_social.max() - site_social.min())
    v_site_majority = float(site_majority.max() - site_majority.min())

    # ----- Variation across developmental stages (age) -----
    # Use quartile-based age bins to capture developmental differences
    age_bins_all = pd.qcut(df["feature3"], 4, duplicates="drop")
    age_social = df.groupby(age_bins_all)["social_use"].mean()

    age_bins_social = pd.qcut(social_users["feature3"], 4, duplicates="drop")
    age_majority = social_users.groupby(age_bins_social)["majority_choice"].mean()

    v_age_social = float(age_social.max() - age_social.min())
    v_age_majority = float(age_majority.max() - age_majority.min())

    # Aggregate variation metrics
    metrics = np.array(
        [v_site_social, v_site_majority, v_age_social, v_age_majority], dtype=float
    )
    v_total = float(metrics.mean())

    # Map variation magnitude to Likert-style scalar in [-100, 100]
    # Heuristic: v_total ≈ 0.10 treated as neutral (0),
    #            v_total ≥ 0.30 treated as very strong variation (+100),
    #            v_total ≈ 0.00 treated as moderate evidence of no variation (about -50).
    center = 0.10
    scale = 0.20  # 0.10 ± 0.20 → [-100, 100] before clipping
    score = (v_total - center) / scale
    scalar = int(np.clip(round(score * 100), -100, 100))

    # Write final scalar conclusion
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        # Single integer value only, no extra text
        f.write(str(scalar))


if __name__ == "__main__":
    main()

