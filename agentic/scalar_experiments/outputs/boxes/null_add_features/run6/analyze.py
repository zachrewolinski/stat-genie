import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data():
    base = Path(__file__).parent
    info_path = base / "info.json"
    data_path = base / "boxes.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)
    return info, df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Majority choice indicator: 1 if child followed the majority, 0 otherwise.
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Basic sanity filters: keep rows with non-missing core fields.
    core_cols = ["majority_choice", "age", "culture"]
    df = df.dropna(subset=core_cols)

    # Age as numeric and define within-sample developmental stages via quartiles.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"])

    # Create four age bins (developmental stages) based on the empirical distribution.
    df["age_stage"] = pd.qcut(df["age"], q=4, labels=False, duplicates="drop")

    return df


def summarize_variation(df: pd.DataFrame):
    # Majority choice rate by culture.
    culture_summary = (
        df.groupby("culture")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .reset_index()
    )

    # Majority choice rate by age stage.
    age_stage_summary = (
        df.groupby("age_stage")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .reset_index()
    )

    # Continuous association between age and majority choice.
    if df["age"].nunique() > 1:
        age_corr = df[["age", "majority_choice"]].corr().iloc[0, 1]
    else:
        age_corr = 0.0

    culture_range = float(culture_summary["majority_rate"].max() - culture_summary["majority_rate"].min())
    age_stage_range = float(age_stage_summary["majority_rate"].max() - age_stage_summary["majority_rate"].min())

    return {
        "culture_summary": culture_summary,
        "age_stage_summary": age_stage_summary,
        "age_corr": age_corr,
        "culture_range": culture_range,
        "age_stage_range": age_stage_range,
    }


def compute_scalar_conclusion(culture_range: float, age_stage_range: float, age_corr: float) -> int:
    """
    Map the strength of variation across cultures and age stages
    to a single Likert-style scalar in [-100, 100].
    Positive values indicate evidence that reliance on social
    information / majority cues DOES vary by culture and development.
    Negative values indicate evidence that it does NOT meaningfully vary.
    """

    # Contribution from culture differences in majority choice rates.
    if culture_range >= 0.25:
        culture_score = 45
    elif culture_range >= 0.15:
        culture_score = 30
    elif culture_range >= 0.05:
        culture_score = 15
    else:
        culture_score = 0

    # Contribution from age-stage differences in majority choice rates.
    if age_stage_range >= 0.25:
        age_stage_score = 35
    elif age_stage_range >= 0.15:
        age_stage_score = 25
    elif age_stage_range >= 0.05:
        age_stage_score = 10
    else:
        age_stage_score = 0

    # Additional small contribution from the continuous age correlation.
    corr_abs = abs(age_corr)
    if corr_abs >= 0.25:
        corr_score = 10
    elif corr_abs >= 0.15:
        corr_score = 5
    else:
        corr_score = 0

    score = culture_score + age_stage_score + corr_score

    # If there is essentially no variation anywhere, treat as evidence for "no".
    if culture_range < 0.05 and age_stage_range < 0.05 and corr_abs < 0.05:
        score = -20

    # Clamp to the required range and return as an integer.
    score = max(-100, min(100, score))
    return int(round(score))


def main():
    info, df = load_data()
    df = prepare_variables(df)
    summary = summarize_variation(df)

    culture_range = summary["culture_range"]
    age_stage_range = summary["age_stage_range"]
    age_corr = summary["age_corr"]

    scalar = compute_scalar_conclusion(culture_range, age_stage_range, age_corr)

    # Write scalar conclusion to file as required (single integer, no extra text).
    conclusion_path = Path(__file__).parent / "conclusion.txt"
    conclusion_path.write_text(f"{scalar}", encoding="utf-8")

    # Optional: print a compact diagnostic summary to stdout for human inspection.
    print("Research question:", info["research_questions"][0])
    print(f"Culture range in majority rates: {culture_range:.3f}")
    print(f"Age-stage range in majority rates: {age_stage_range:.3f}")
    print(f"Correlation(age, majority_choice): {age_corr:.3f}")
    print(f"Scalar conclusion (Likert -100..100): {scalar}")


if __name__ == "__main__":
    main()

