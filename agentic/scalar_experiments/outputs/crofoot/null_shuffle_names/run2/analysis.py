import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data():
    df = pd.read_csv("crofoot.csv")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Outcome: 1 if focal group won, 0 otherwise
    df = df.copy()
    df["win_focal"] = df["m_focal"]

    # Group sizes (see info.json descriptions)
    focal_size = df["f_other"]  # "Number of individuals in focal group"
    other_size = df["win"]  # "Number of individuals in other group"
    df["rel_size"] = focal_size - other_size
    df["rel_size_ratio"] = focal_size / other_size
    df["focal_larger"] = (df["rel_size"] > 0).astype(int)
    df["focal_smaller"] = (df["rel_size"] < 0).astype(int)

    # Contest location: distance of each group from its home range center
    dist_focal_home = df["m_other"]  # distance of focal from its home range center
    dist_other_home = df["n_focal"]  # distance of other from its home range center
    df["rel_dist"] = dist_other_home - dist_focal_home
    # Positive rel_dist -> focal is closer to its own center than other is to its own
    df["closer_to_focal_core"] = (df["rel_dist"] > 0).astype(int)

    return df


def summarize_effects(df: pd.DataFrame) -> dict:
    results = {}

    # Win rates by relative size
    larger = df[df["rel_size"] > 0]["win_focal"]
    equal = df[df["rel_size"] == 0]["win_focal"]
    smaller = df[df["rel_size"] < 0]["win_focal"]

    results["win_rate_larger"] = larger.mean() if len(larger) else np.nan
    results["win_rate_equal"] = equal.mean() if len(equal) else np.nan
    results["win_rate_smaller"] = smaller.mean() if len(smaller) else np.nan

    # Win rates by contest location (relative to home range centers)
    near_focal = df[df["closer_to_focal_core"] == 1]["win_focal"]
    near_other = df[df["closer_to_focal_core"] == 0]["win_focal"]

    results["win_rate_near_focal"] = near_focal.mean() if len(near_focal) else np.nan
    results["win_rate_near_other"] = near_other.mean() if len(near_other) else np.nan

    return results


def logistic_analysis(df: pd.DataFrame) -> dict:
    """Fit a simple logistic regression for interpretability."""
    # Use standardized predictors for stability
    X = df[["rel_size", "rel_dist"]].astype(float)
    X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)
    y = df["win_focal"].astype(float)

    try:
        model = sm.Logit(y, X, missing="drop")
        fit = model.fit(disp=False)
    except Exception:
        return {
            "coef_rel_size": np.nan,
            "p_rel_size": np.nan,
            "coef_rel_dist": np.nan,
            "p_rel_dist": np.nan,
            "pseudo_r2": np.nan,
        }

    params = fit.params
    pvalues = fit.pvalues
    pseudo_r2 = 1 - fit.llf / fit.llnull if fit.llnull != 0 else np.nan

    return {
        "coef_rel_size": params.get("rel_size", np.nan),
        "p_rel_size": pvalues.get("rel_size", np.nan),
        "coef_rel_dist": params.get("rel_dist", np.nan),
        "p_rel_dist": pvalues.get("rel_dist", np.nan),
        "pseudo_r2": pseudo_r2,
    }


def map_to_scalar(summary: dict, logit: dict) -> int:
    """
    Map the strength of evidence that relative group size and contest location
    influence win probability to a Likert-style scalar in [-100, 100].
    """
    evidence_score = 0.0

    # 1) Descriptive effect of relative size
    wr_larger = summary.get("win_rate_larger")
    wr_smaller = summary.get("win_rate_smaller")
    if not np.isnan(wr_larger) and not np.isnan(wr_smaller):
        diff = wr_larger - wr_smaller
        # Scale 0..1 difference to 0..40 points
        evidence_score += 40 * max(0.0, min(1.0, abs(diff)))

    # 2) Descriptive effect of location
    wr_near_focal = summary.get("win_rate_near_focal")
    wr_near_other = summary.get("win_rate_near_other")
    if not np.isnan(wr_near_focal) and not np.isnan(wr_near_other):
        diff_loc = wr_near_focal - wr_near_other
        evidence_score += 30 * max(0.0, min(1.0, abs(diff_loc)))

    # 3) Logistic regression significance
    p_size = logit.get("p_rel_size", np.nan)
    p_dist = logit.get("p_rel_dist", np.nan)
    pseudo_r2 = logit.get("pseudo_r2", np.nan)

    for p in (p_size, p_dist):
        if np.isnan(p):
            continue
        if p < 0.01:
            evidence_score += 15
        elif p < 0.05:
            evidence_score += 10
        elif p < 0.1:
            evidence_score += 5

    if not np.isnan(pseudo_r2):
        # Pseudo R^2 up to ~0.4 is typical; scale modestly
        evidence_score += 20 * max(0.0, min(0.4, pseudo_r2)) / 0.4

    # Cap in [0, 100] and convert to an integer in [0, 100]
    evidence_score = max(0.0, min(100.0, evidence_score))

    # The question is whether the factors influence win probability.
    # Evidence only moves us towards "Yes", never towards "No" with this dataset.
    scalar = int(round(evidence_score))
    return scalar


def main():
    df = load_data()
    df_feat = engineer_features(df)

    summary = summarize_effects(df_feat)
    logit = logistic_analysis(df_feat)

    scalar = map_to_scalar(summary, logit)

    # Persist scalar conclusion
    Path("conclusion.txt").write_text(str(scalar), encoding="utf-8")

    # Optional: print a brief JSON summary for human inspection (does not go into conclusion.txt)
    debug = {
        "summary": summary,
        "logit": logit,
        "scalar": scalar,
    }
    print(json.dumps(debug, indent=2, default=float))


if __name__ == "__main__":
    main()

