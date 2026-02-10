import json
from pathlib import Path

import numpy as np
import pandas as pd

INFO_PATH = Path("info.json")
CSV_PATH = Path("boxes.csv")


def load_data():
    info = json.loads(INFO_PATH.read_text())
    df = pd.read_csv(CSV_PATH)
    return info, df


def compute_metrics(df: pd.DataFrame):
    # feature1: 1=unchosen option, 2=majority option, 3=minority option
    outcome = df["feature1"]
    n = len(outcome)
    majority_rate = (outcome == 2).mean()
    minority_rate = (outcome == 3).mean()
    undemo_rate = (outcome == 1).mean()

    # Age and site
    ages = df["feature3"]
    sites = df["feature5"]

    # Age trend in majority choice: correlation
    age_majority_corr = np.corrcoef(ages, (outcome == 2).astype(float))[0, 1]

    # Site-level variation in majority choice
    site_majority = df.groupby("feature5")["feature1"].apply(lambda s: (s == 2).mean())
    site_variation = site_majority.max() - site_majority.min()

    metrics = {
        "n": int(n),
        "majority_rate": float(majority_rate),
        "minority_rate": float(minority_rate),
        "undemo_rate": float(undemo_rate),
        "age_majority_corr": float(age_majority_corr),
        "site_majority_min": float(site_majority.min()),
        "site_majority_max": float(site_majority.max()),
        "site_variation": float(site_variation),
    }
    return metrics


def derive_scalar(metrics: dict) -> int:
    # Base signal: overall reliance on majority as form of social information
    majority = metrics["majority_rate"]
    minority = metrics["minority_rate"]
    undemo = metrics["undemo_rate"]

    # Strong evidence of reliance on social information if majority + minority >> undemonstrated
    social_reliance = (majority + minority) - undemo

    # Normalize to [-1, 1] using a simple bounded transform
    # social_reliance is in [-1, 1]; map linearly
    social_component = social_reliance

    # Majority preference component: majority vs minority
    if majority + minority > 0:
        majority_bias = (majority - minority) / (majority + minority)
    else:
        majority_bias = 0.0

    # Age trend: correlation already in [-1,1]
    age_trend = metrics["age_majority_corr"]

    # Cross-cultural variation: large variation means majority preference is present but context-dependent.
    # We treat moderate variation as supporting the idea that reliance and preference vary across cultures.
    variation = metrics["site_variation"]  # in [0,1]
    variation_component = 2 * (variation - 0.25)  # center ~0.25, scale to roughly [-0.5, 1.5]

    # Combine components with heuristic weights reflecting the question
    score = (
        0.3 * social_component
        + 0.25 * majority_bias
        + 0.25 * age_trend
        + 0.2 * variation_component
    )

    # Clip to [-1,1] and rescale to [-100,100]
    score = max(-1.0, min(1.0, score))
    scalar = int(round(score * 100))
    return scalar


def main():
    info, df = load_data()
    metrics = compute_metrics(df)

    scalar = derive_scalar(metrics)

    # For debugging during development we could print metrics, but per instructions
    # the final output file must contain only the scalar.
    Path("conclusion.txt").write_text(str(scalar))


if __name__ == "__main__":
    main()
