import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("boxes.csv")

    # Outcome coding: 1 = unchosen, 2 = majority, 3 = minority
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df["minority_choice"] = (df["feature1"] == 3).astype(int)

    # Basic summaries
    n = len(df)
    majority_rate = df["majority_choice"].mean()
    minority_rate = df["minority_choice"].mean()

    print(f"N = {n}")
    print(f"Overall majority choice rate: {majority_rate:.3f}")
    print(f"Overall minority choice rate: {minority_rate:.3f}")

    # Age effect: majority choice by age in years
    age_summary = (
        df.groupby("feature3")["majority_choice"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "majority_rate", "count": "n"})
    )
    print("\nMajority choice by age (feature3 = age):")
    print(age_summary)

    # Site / culture effect: majority choice by site ID
    site_summary = (
        df.groupby("feature5")["majority_choice"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "majority_rate", "count": "n"})
    )
    print("\nMajority choice by site (feature5 = site id):")
    print(site_summary)

    # Simple measures of variability across age and site
    age_var_range = age_summary["majority_rate"].max() - age_summary["majority_rate"].min()
    site_var_range = site_summary["majority_rate"].max() - site_summary["majority_rate"].min()

    print(f"\nRange of majority rates across ages: {age_var_range:.3f}")
    print(f"Range of majority rates across sites: {site_var_range:.3f}")

    # Heuristic scalar mapping:
    # We interpret larger variability in majority choices across age and site
    # as stronger evidence that reliance on social information / majority cues varies.
    #
    # Combine age and site variability into a single measure.
    combined_var = (age_var_range + site_var_range) / 2
    print(f"Combined variability measure: {combined_var:.3f}")

    # Map combined variability in [0, 1] approximately to a Likert scale [-100, 100].
    # Clip to [0, 1] to be safe.
    combined_clipped = float(np.clip(combined_var, 0.0, 1.0))
    scalar = int(round(200 * combined_clipped - 100))

    print(f"\nChosen scalar (Likert -100 to 100): {scalar}")

    # Write scalar to conclusion.txt with no extra text.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

