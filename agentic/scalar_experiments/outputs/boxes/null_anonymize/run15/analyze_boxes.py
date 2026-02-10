import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_metadata():
    info_path = Path("info.json")
    with info_path.open("r") as f:
        return json.load(f)


def load_data():
    return pd.read_csv("boxes.csv")


def compute_overall_majority_reliance(df: pd.DataFrame) -> float:
    """
    Proportion of trials where children follow the majority option (feature1 == 2).
    """
    return (df["feature1"] == 2).mean()


def majority_reliance_by_age(df: pd.DataFrame):
    df = df.copy()
    df["is_majority"] = (df["feature1"] == 2).astype(int)
    # Age is discrete 4-14; treat it as continuous for a simple trend check.
    model = smf.logit("is_majority ~ feature3", data=df).fit(disp=False)
    return model


def majority_reliance_by_site(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["is_majority"] = (df["feature1"] == 2).astype(int)
    return df.groupby("feature5")["is_majority"].mean()


def summarize_evidence(df: pd.DataFrame) -> int:
    """
    Convert statistical evidence into an integer Likert score from -100 to 100
    answering:
    \"Do children's reliance on social information and preference for majority cues
    vary across cultures and developmental stages?\"

    Positive values reflect evidence that both age and culture matter,
    negative that they clearly do not, and values near zero indicate weak or
    ambiguous evidence.
    """
    overall_majority = compute_overall_majority_reliance(df)

    # Age effect via logistic regression.
    age_model = majority_reliance_by_age(df)
    age_p = age_model.pvalues.get("feature3", 1.0)
    age_coef = age_model.params.get("feature3", 0.0)

    # Cultural variation via between-site variability in majority following.
    site_means = majority_reliance_by_site(df)
    site_range = site_means.max() - site_means.min()
    site_std = site_means.std()

    # Simple heuristic scoring:
    # - Strong majority bias overall supports the premise that children do use
    #   majority cues (baseline positive weight).
    # - Strong age effect (small p-value, sizable coefficient) increases score.
    # - Large cross-site variability also increases score.
    score = 0.0

    # Baseline: if most children follow majority, acknowledge robust majority use.
    if overall_majority > 0.6:
        score += 25
    elif overall_majority > 0.5:
        score += 10
    elif overall_majority < 0.4:
        score -= 10

    # Age trend contribution.
    if age_p < 0.001 and abs(age_coef) > 0.1:
        score += 40 * np.sign(age_coef)
    elif age_p < 0.01 and abs(age_coef) > 0.05:
        score += 25 * np.sign(age_coef)
    elif age_p < 0.05:
        score += 10 * np.sign(age_coef)
    else:
        # Evidence that age does not matter much.
        score += 0

    # Cultural (site) variation contribution.
    # Range up to 1.0; std on proportions typically <= 0.3 here.
    if site_range > 0.3 or (site_std is not None and site_std > 0.12):
        score += 25
    elif site_range > 0.15 or (site_std is not None and site_std > 0.06):
        score += 10
    else:
        score += 0

    # Clip to [-100, 100] and round to nearest integer.
    score = int(np.clip(round(score), -100, 100))
    return score


def main():
    _ = load_metadata()  # Loaded for context; analysis uses encoded columns.
    df = load_data()
    score = summarize_evidence(df)
    # Write scalar only to conclusion.txt as required.
    Path("conclusion.txt").write_text(str(score), encoding="utf-8")


if __name__ == "__main__":
    main()

