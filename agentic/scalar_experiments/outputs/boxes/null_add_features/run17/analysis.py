import pathlib

import numpy as np
import pandas as pd


def main() -> None:
    data_path = pathlib.Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Keep only valid outcome codes.
    df = df[df["y"].isin([1, 2, 3])].copy()

    # Social-information use: any demonstrated option (majority or minority).
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    social_df = df[df["social"] == 1].copy()
    social_df = social_df.dropna(subset=["majority_choice"])

    # Variation in majority preference across cultures.
    if "culture" in social_df.columns and social_df["culture"].nunique() > 1:
        culture_rates = social_df.groupby("culture")["majority_choice"].mean()
        v_culture = float(culture_rates.max() - culture_rates.min())
    else:
        v_culture = 0.0

    # Variation in majority preference across age groups (developmental stages).
    if "age" in social_df.columns and social_df["age"].nunique() > 1:
        n_bins = min(4, social_df["age"].nunique())
        social_df["age_bin"] = pd.cut(social_df["age"], bins=n_bins, include_lowest=True)
        age_rates = social_df.groupby("age_bin")["majority_choice"].mean()
        if age_rates.size > 1:
            v_age = float(age_rates.max() - age_rates.min())
        else:
            v_age = 0.0
    else:
        v_age = 0.0

    # Combine culture and age variation into a single index.
    v_total = float(np.mean([v_culture, v_age]))

    # Map variation index to Likert scale [-100, 100]:
    # - Threshold of 0.05: below this, treated as "no meaningful variation".
    # - Scale of 0.25: values around 0.3 or higher map near 100 ("strong yes").
    threshold = 0.05
    scale = 0.25
    norm = (v_total - threshold) / scale
    scalar = int(np.clip(round(norm * 100), -100, 100))

    # Write only the scalar value to conclusion.txt (no extra text or lines).
    conclusion_path = pathlib.Path("conclusion.txt")
    conclusion_path.write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

