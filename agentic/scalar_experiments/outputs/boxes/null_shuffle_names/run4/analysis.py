import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent

def load_metadata():
    info_path = HERE / "info.json"
    with info_path.open("r") as f:
        return json.load(f)


def load_data():
    csv_path = HERE / "boxes.csv"
    df = pd.read_csv(csv_path)
    return df


def compute_reliance_indices(df: pd.DataFrame):
    # Map outcome codes: 1 = undemonstrated, 2 = majority, 3 = minority
    majority = (df["majority_first"] == 2).astype(int)
    minority = (df["majority_first"] == 3).astype(int)
    undemo = (df["majority_first"] == 1).astype(int)

    overall_n = len(df)
    overall_majority_rate = majority.mean()
    overall_minority_rate = minority.mean()
    overall_undemo_rate = undemo.mean()

    # Age-wise majority preference (by year)
    age_group = df["age"]
    age_stats = (
        df.assign(majority=majority, minority=minority, undemo=undemo)
        .groupby("age")[["majority", "minority", "undemo"]]
        .mean()
    )

    # Culture-wise majority preference (using y as site ID)
    site_stats = (
        df.assign(majority=majority, minority=minority, undemo=undemo)
        .groupby("y")[["majority", "minority", "undemo"]]
        .mean()
    )

    return {
        "overall_n": int(overall_n),
        "overall_majority_rate": float(overall_majority_rate),
        "overall_minority_rate": float(overall_minority_rate),
        "overall_undemo_rate": float(overall_undemo_rate),
        "age_stats": age_stats,
        "site_stats": site_stats,
    }


def summarize_development(age_stats: pd.DataFrame):
    # Measure linear association between age and majority choice
    ages = age_stats.index.to_numpy(dtype=float)
    majority_rates = age_stats["majority"].to_numpy(dtype=float)

    if len(ages) < 2:
        return {"age_corr": 0.0, "age_slope": 0.0}

    age_corr = float(np.corrcoef(ages, majority_rates)[0, 1])

    # Simple linear regression slope
    A = np.vstack([ages, np.ones_like(ages)]).T
    slope, _ = np.linalg.lstsq(A, majority_rates, rcond=None)[0]

    return {"age_corr": age_corr, "age_slope": float(slope)}


def summarize_cross_culture(site_stats: pd.DataFrame):
    # Variation in majority rates across sites as a proxy for cultural variability
    majority_rates = site_stats["majority"].to_numpy(dtype=float)
    if len(majority_rates) == 0:
        return {"mean_majority": 0.0, "std_majority": 0.0}

    return {
        "mean_majority": float(np.mean(majority_rates)),
        "std_majority": float(np.std(majority_rates, ddof=0)),
    }


def map_to_scalar(overall_majority_rate: float, dev: dict, cross: dict) -> int:
    """Map empirical patterns to Likert scalar [-100, 100].

    Intuition:
    - Strong overall majority preference -> positive evidence for reliance on social/majority info.
    - Positive age slope/correlation -> developmental increase in majority reliance.
    - Some cross-cultural variation but majority-bias present across sites -> nuanced but overall "Yes".
    """

    # Base on overall majority preference: 0.5 ~ neutral, 1.0 ~ strong yes
    base = (overall_majority_rate - 0.5) * 200  # 0.5->0, 1.0->100, 0.0->-100

    # Age trend: boost if majority reliance increases with age
    age_boost = 0.0
    if dev["age_slope"] > 0:
        age_boost += min(15.0, dev["age_slope"] * 200)
    elif dev["age_slope"] < 0:
        age_boost += max(-15.0, dev["age_slope"] * 200)

    # Cross-cultural variability: if std is large, slightly dampen extremity
    std_m = cross["std_majority"]
    damp_factor = 1.0
    if std_m > 0.15:
        damp_factor = 0.8
    elif std_m > 0.25:
        damp_factor = 0.7

    raw_score = (base + age_boost) * damp_factor
    # Clip to [-100, 100] and round to nearest integer
    raw_score = max(-100.0, min(100.0, raw_score))
    return int(round(raw_score))


def main():
    _meta = load_metadata()
    df = load_data()

    indices = compute_reliance_indices(df)
    dev = summarize_development(indices["age_stats"])
    cross = summarize_cross_culture(indices["site_stats"])

    scalar = map_to_scalar(indices["overall_majority_rate"], dev, cross)

    out_path = HERE / "conclusion.txt"
    out_path.write_text(str(scalar))


if __name__ == "__main__":
    main()
