import pandas as pd
import numpy as np


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Operationalize key constructs
    # Reliance on social information: chose majority or minority option (y != 1)
    df["used_social"] = (df["y"] != 1).astype(int)

    # Preference for majority cues: among social choices, chose majority option (y == 2)
    df_social = df[df["used_social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Age groups (developmental stages) via quartiles
    df["age_group"] = pd.qcut(df["age"], q=4, labels=False, duplicates="drop")
    df_social = df_social.merge(
        df[["age", "age_group"]], on=["age"], how="left", suffixes=("", "_grp")
    )

    # Variation across age groups
    age_used = (
        df.groupby("age_group", dropna=True)["used_social"]
        .mean()
        .dropna()
    )
    age_majority = (
        df_social.groupby("age_group", dropna=True)["majority_choice"]
        .mean()
        .dropna()
    )

    var_age_used = float(age_used.max() - age_used.min()) if not age_used.empty else 0.0
    var_age_majority = (
        float(age_majority.max() - age_majority.min())
        if not age_majority.empty
        else 0.0
    )

    # Variation across cultural groups
    culture_used = (
        df.groupby("culture", dropna=True)["used_social"]
        .mean()
        .dropna()
    )

    culture_majority = (
        df_social.groupby("culture", dropna=True)["majority_choice"]
        .mean()
        .dropna()
    )

    var_culture_used = (
        float(culture_used.max() - culture_used.min())
        if not culture_used.empty
        else 0.0
    )
    var_culture_majority = (
        float(culture_majority.max() - culture_majority.min())
        if not culture_majority.empty
        else 0.0
    )

    # Aggregate variation metrics into a single Likert-style scalar (0–100)
    variations = np.array(
        [
            var_age_used,
            var_age_majority,
            var_culture_used,
            var_culture_majority,
        ],
        dtype=float,
    )

    avg_var = float(variations.mean()) if variations.size > 0 else 0.0

    # Map average range in probabilities (0–1) to 0–100 scale
    score = int(round(100 * avg_var))

    # Ensure scalar is within [-100, 100] as required
    score = max(-100, min(100, score))

    with open("conclusion.txt", "w") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()

