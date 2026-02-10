import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic sanity checks
    assert {"y", "gender", "age", "majority_first", "culture"}.issubset(df.columns)

    # Encode majority choice vs. all other choices
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Center age for stability and create age groupings
    df["age_c"] = df["age"] - df["age"].mean()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    # Descriptive majority-choice rates by age group and culture
    desc_age = (
        df.groupby("age_group")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .reset_index()
    )
    desc_culture = (
        df.groupby("culture")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .reset_index()
    )

    print("Majority-choice rate by age group:")
    print(desc_age.to_string(index=False))
    print("\nMajority-choice rate by culture:")
    print(desc_culture.to_string(index=False))

    # Logistic regression: majority choice ~ age + culture + majority_first
    # Treat culture as categorical to capture cross-cultural variation.
    model = smf.glm(
        formula="majority_choice ~ age_c + C(culture) + majority_first",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    print("\nLogistic regression summary (majority_choice ~ age + culture + majority_first):")
    print(model.summary())

    # Extract evidence strength:
    # - age effect: absolute z-stat for age_c
    # - joint culture effects: mean absolute z of culture indicators
    params = model.params
    bse = model.bse
    z_stats = params / bse

    age_z = float(abs(z_stats.get("age_c", 0.0)))
    culture_zs = [
        abs(z_stats[name])
        for name in z_stats.index
        if name.startswith("C(culture)[T.")
    ]
    culture_z_mean = float(np.mean(culture_zs)) if culture_zs else 0.0

    print(f"\nAge effect |z|: {age_z:.3f}")
    print(f"Mean |z| across culture indicators: {culture_z_mean:.3f}")

    # Map combined evidence to a Likert-style scalar in [-100, 100],
    # emphasizing that strong, consistent effects support a strong "Yes".
    # We cap the combined z-score at 6 for stability.
    combined_z = age_z + culture_z_mean
    capped = min(combined_z, 6.0)
    scalar = int(round((capped / 6.0) * 100))

    # Ensure scalar is within bounds
    scalar = max(-100, min(100, scalar))

    print(f"\nDerived scalar conclusion (Likert -100 to 100): {scalar}")

    # Write scalar only to conclusion.txt as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

