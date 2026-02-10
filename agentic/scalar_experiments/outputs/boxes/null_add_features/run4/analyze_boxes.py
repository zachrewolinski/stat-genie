import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_scalar_from_deltas(delta_culture: float, delta_age: float) -> int:
    """
    Map effect size-style deltas to a Likert scalar in [-100, 100].

    - delta_culture: range of majority-choice proportions across cultures.
    - delta_age: change in majority-choice probability between age Q1 and Q3.
    """
    dc = max(0.0, min(1.0, float(delta_culture)))
    da = max(0.0, min(1.0, float(delta_age)))

    # If both deltas are extremely small, treat as strong evidence of (practically)
    # no meaningful variation.
    if dc < 0.03 and da < 0.02:
        return -60

    # Normalize deltas relative to "large" benchmarks:
    # ~0.30 range across cultures and ~0.15 change across age considered strong.
    dc_norm = min(dc / 0.30, 1.0)
    da_norm = min(da / 0.15, 1.0)

    # Weighted combination, emphasizing cross-cultural differences slightly more.
    combined = 0.6 * dc_norm + 0.4 * da_norm  # in [0, 1]

    # Piecewise mapping from combined evidence to Likert strength.
    if combined < 0.15:
        scalar = 0
    elif combined < 0.30:
        scalar = 30
    elif combined < 0.50:
        scalar = 50
    elif combined < 0.70:
        scalar = 70
    else:
        scalar = 90

    return int(scalar)


def main() -> None:
    data_path = Path("boxes.csv")
    if not data_path.exists():
        raise FileNotFoundError("boxes.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Outcome coding: 1 = undemonstrated third option, 2 = majority, 3 = minority.
    df["majority"] = (df["y"] == 2).astype(int)
    df["social"] = (df["y"] != 1).astype(int)

    # --- Cross-cultural variation: majority choice by culture ---
    culture_group = df.groupby("culture")["majority"].mean()
    delta_culture = float(culture_group.max() - culture_group.min())

    # --- Developmental (age) variation: logistic model and predicted change ---
    # Guard against degenerate age data.
    if df["age"].nunique() > 1:
        age_q1, age_q3 = df["age"].quantile([0.25, 0.75])

        glm_age = smf.glm(
            formula="majority ~ age",
            data=df,
            family=sm.families.Binomial(),
        ).fit()

        age_pred_df = pd.DataFrame({"age": [age_q1, age_q3]})
        age_pred_probs = glm_age.predict(age_pred_df)
        delta_age = float(abs(age_pred_probs.iloc[1] - age_pred_probs.iloc[0]))
    else:
        delta_age = 0.0

    # Compute scalar conclusion from the two deltas.
    scalar = compute_scalar_from_deltas(delta_culture, delta_age)

    # Print a brief analysis summary to stdout (for human inspection).
    overall_majority = df["majority"].mean()
    overall_social = df["social"].mean()

    print("Overall reliance on social information (any demonstrator):")
    print(f"  Mean social choice rate: {overall_social:.3f}")
    print("Overall preference for majority option:")
    print(f"  Mean majority choice rate: {overall_majority:.3f}")
    print("\nCross-cultural variation in majority choice:")
    print(culture_group.to_frame(name="majority_rate"))
    print(f"  Range (max - min) across cultures: {delta_culture:.3f}")

    print("\nDevelopmental (age) variation in majority preference:")
    print(f"  Predicted change in majority probability from age Q1 to Q3: {delta_age:.3f}")

    print("\nDerived Likert-style scalar (−100 to 100, higher = stronger 'Yes'):")
    print(f"  Scalar: {scalar}")

    # Write scalar to conclusion.txt with no extra text.
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(f"{scalar}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

