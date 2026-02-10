import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_majority_preference(df: pd.DataFrame) -> dict:
    # majority_first: 1 = unchosen, 2 = majority, 3 = minority
    total_n = len(df)
    majority_n = (df["majority_first"] == 2).sum()
    minority_n = (df["majority_first"] == 3).sum()
    unchosen_n = (df["majority_first"] == 1).sum()

    return {
        "total_n": int(total_n),
        "majority_n": int(majority_n),
        "minority_n": int(minority_n),
        "unchosen_n": int(unchosen_n),
        "majority_rate": float(majority_n / total_n) if total_n > 0 else np.nan,
        "minority_rate": float(minority_n / total_n) if total_n > 0 else np.nan,
        "unchosen_rate": float(unchosen_n / total_n) if total_n > 0 else np.nan,
    }


def compute_developmental_trend(df: pd.DataFrame) -> float:
    """
    Approximate developmental trend:
    correlation between age and majority choice (binary: 1 if majority, 0 otherwise).
    """
    df = df.copy()
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    if df["age"].nunique() < 2:
        return 0.0
    return float(df["age"].corr(df["majority_choice"]))


def compute_cultural_variation(df: pd.DataFrame) -> float:
    """
    Approximate cultural variation via spread of site-level majority rates.
    y is site ID (1-8). Higher std implies stronger variation.
    """
    site_rates = (
        df.groupby("y")["majority_first"]
        .apply(lambda s: (s == 2).mean())
        .values
    )
    if len(site_rates) <= 1:
        return 0.0
    return float(np.std(site_rates))


def scalar_from_effects(
    majority_rate: float, dev_trend: float, cultural_var: float
) -> int:
    """
    Map observed patterns to Likert scale:
    - Strong overall majority use
    - Positive developmental trend
    - Non-trivial cultural variation
    together support a "Yes" answer.
    """
    score = 0.0

    # Baseline from overall majority preference
    # Center 0.5 as neutral; each 0.1 above adds 10 points.
    score += (majority_rate - 0.5) * 100

    # Developmental trend: correlation in [-1,1], weight moderately.
    score += dev_trend * 30

    # Cultural variation: std in [0, 0.5] roughly; scale to [-20,20] but only positive variation increases support.
    score += max(cultural_var, 0.0) * 40

    # Clip to [-100, 100] and round to nearest int.
    score = max(-100.0, min(100.0, score))
    return int(round(score))


def main() -> None:
    base = Path(".")
    meta = load_metadata(base / "info.json")
    df = load_data(base / "boxes.csv")

    summary = compute_majority_preference(df)
    dev_trend = compute_developmental_trend(df)
    cultural_var = compute_cultural_variation(df)

    scalar = scalar_from_effects(
        summary["majority_rate"], dev_trend, cultural_var
    )

    # Write scalar only to conclusion.txt as required.
    (base / "conclusion.txt").write_text(str(scalar), encoding="utf-8")


if __name__ == "__main__":
    main()

