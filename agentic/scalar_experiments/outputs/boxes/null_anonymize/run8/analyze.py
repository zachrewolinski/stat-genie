import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(csv_path: str = "boxes.csv") -> pd.DataFrame:
    """Load the dataset."""
    df = pd.read_csv(csv_path)
    return df


def compute_effect_measures(df: pd.DataFrame) -> dict:
    """
    Compute simple effect-size style measures relevant to the research question:
    - Variation in majority choice across developmental stages (age groups).
    - Variation in majority choice across cultural sites.
    - Correlation between age and majority choice.
    """
    # Majority choice indicator: feature1 == 2
    df = df.copy()
    df["is_majority"] = df["feature1"] == 2

    # Age variable
    age = df["feature3"].astype(float)

    # Developmental stages via age tertiles
    try:
        age_bins = pd.qcut(age, 3, labels=["young", "middle", "old"])
    except ValueError:
        # Fallback: simple bins if qcut fails (e.g., too few unique values)
        age_bins = pd.cut(
            age,
            bins=[age.min() - 1e-6, 7, 11, age.max() + 1e-6],
            labels=["young", "middle", "old"],
            include_lowest=True,
        )

    df["age_group"] = age_bins

    age_group_means = (
        df.groupby("age_group")["is_majority"].mean().dropna()
        if "age_group" in df
        else pd.Series(dtype=float)
    )
    age_diff = float(age_group_means.max() - age_group_means.min()) if not age_group_means.empty else 0.0

    # Cultural sites via feature5
    site_means = df.groupby("feature5")["is_majority"].mean()
    site_diff = float(site_means.max() - site_means.min()) if not site_means.empty else 0.0

    # Correlation between age and majority choice
    majority_numeric = df["is_majority"].astype(int)
    if majority_numeric.std(ddof=0) == 0 or age.std(ddof=0) == 0:
        corr_age = 0.0
    else:
        corr_age = float(age.corr(majority_numeric))

    return {
        "age_diff": age_diff,
        "site_diff": site_diff,
        "corr_age": corr_age,
    }


def map_evidence_to_scalar(age_diff: float, site_diff: float, corr_age: float) -> int:
    """
    Map effect measures to a Likert-style scalar in [-100, 100]
    answering: "Do children's reliance on social information and
    preference for majority cues vary across cultures and developmental stages?"

    Positive values indicate evidence that they DO vary (Yes),
    negative values indicate evidence that they DO NOT vary (No),
    and values near zero indicate ambiguous/neutral evidence.
    """
    age_diff_abs = abs(age_diff)
    site_diff_abs = abs(site_diff)
    corr_abs = abs(corr_age)

    # Strong evidence of *no* variation: all metrics extremely small
    if age_diff_abs < 0.02 and site_diff_abs < 0.02 and corr_abs < 0.02:
        return -80

    # Moderate evidence of little/no variation
    if age_diff_abs < 0.05 and site_diff_abs < 0.05 and corr_abs < 0.05:
        return -40

    # Weak evidence of little/no variation
    if age_diff_abs < 0.10 and site_diff_abs < 0.10 and corr_abs < 0.10:
        return -15

    # Otherwise, there is at least some evidence of variation.
    score = 0

    # Contribution from age-group differences in majority choice
    if age_diff_abs > 0.25:
        score += 35
    elif age_diff_abs > 0.10:
        score += 20
    elif age_diff_abs > 0.05:
        score += 10

    # Contribution from cross-cultural (site) differences
    if site_diff_abs > 0.25:
        score += 35
    elif site_diff_abs > 0.10:
        score += 20
    elif site_diff_abs > 0.05:
        score += 10

    # Contribution from continuous age-majority correlation
    if corr_abs > 0.30:
        score += 30
    elif corr_abs > 0.15:
        score += 15
    elif corr_abs > 0.05:
        score += 5

    # If evidence is present but weak, ensure a minimal positive endorsement.
    if score == 0:
        score = 10

    # Clamp to allowable Likert range and ensure integer
    score = int(round(score))
    score = max(-100, min(100, score))
    return score


def main() -> None:
    df = load_data("boxes.csv")
    measures = compute_effect_measures(df)

    scalar = map_evidence_to_scalar(
        measures["age_diff"],
        measures["site_diff"],
        measures["corr_age"],
    )

    # Write ONLY the scalar integer to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(int(scalar)), encoding="utf-8")


if __name__ == "__main__":
    main()

