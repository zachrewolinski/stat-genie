import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_metadata():
    info_path = ROOT / "info.json"
    with info_path.open("r") as f:
        return json.load(f)


def load_data():
    data_path = ROOT / "boxes.csv"
    df = pd.read_csv(data_path)
    return df


def analyze(df: pd.DataFrame):
    """Return a scalar on [-100, 100] answering the research question.

    Research question (negatively framed):
    "I strongly believe the answer to the following question to be 'No'.
    Do children’s reliance on social information and preference for majority
    cues vary across cultures and developmental stages?"

    We operationalise:
    - Reliance on social information / majority preference: choosing y == 2 (majority)
      rather than y == 1 (undemonstrated) or y == 3 (minority).
    - Variation across age: age dependence of majority choice.
    - Variation across cultures: culture dependence of majority choice.

    We quantify both age and culture effects and combine them into a simple
    evidence score in [-100, 100], where positive values support variation
    (answer "Yes") and negative values support invariance (answer "No").
    """

    # Encode majority choice indicator
    df = df.copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # --- Variation across age (developmental stages) ---
    age = df["age"].values.astype(float)
    maj = df["majority_choice"].values.astype(float)

    age_centered = age - age.mean()
    if np.std(age_centered) > 0:
        age_corr = np.corrcoef(age_centered, maj)[0, 1]
    else:
        age_corr = 0.0

    # --- Variation across cultures ---
    # Compute majority-choice rate per culture and its variance
    culture_means = df.groupby("culture")["majority_choice"].mean()
    culture_var = float(culture_means.var(ddof=1)) if len(culture_means) > 1 else 0.0

    # Normalise metrics to roughly [-1, 1]
    # Age correlation is already in [-1, 1].
    age_strength = abs(age_corr)

    # For culture, a variance of ~0.02 already indicates noticeable spread
    culture_strength = min(culture_var / 0.02, 1.0)

    # Combine with equal weight
    combined_strength = 0.5 * age_strength + 0.5 * culture_strength

    # Map to [-100, 100] supporting "Yes" (variation). We then flip sign
    # because the leading statement claims the answer is "No" and we want
    # to quantify support for that statement.
    evidence_for_variation = combined_strength * 100.0

    # If there is strong evidence for variation, that directly contradicts
    # the initial "No" prior, so the scalar for the research question
    # ("Do they vary?") should be positive, while our Likert scale is
    # defined such that -100 is a very strong "No" and +100 is a very
    # strong "Yes".
    scalar_yes = evidence_for_variation

    # Clip and round to nearest integer
    scalar_yes = int(np.clip(np.round(scalar_yes), -100, 100))
    return scalar_yes, {
        "age_corr": float(age_corr),
        "culture_var": culture_var,
        "age_strength": float(age_strength),
        "culture_strength": float(culture_strength),
        "combined_strength": float(combined_strength),
    }


def main():
    _meta = load_metadata()
    df = load_data()
    scalar, details = analyze(df)

    # Write scalar conclusion
    out_path = ROOT / "conclusion.txt"
    with out_path.open("w") as f:
        f.write(str(scalar))

    # Optionally, also log some diagnostics for manual inspection
    diag_path = ROOT / "analysis_diagnostics.json"
    with diag_path.open("w") as f:
        json.dump(details, f, indent=2)


if __name__ == "__main__":
    main()
