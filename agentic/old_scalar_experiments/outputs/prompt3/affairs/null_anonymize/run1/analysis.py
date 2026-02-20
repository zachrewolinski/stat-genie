import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature2": "affairs_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "rating",
        }
    )
    df["has_affair"] = (df["affairs_freq"] > 0).astype(int)
    df = df.dropna(subset=["has_affair", "children"])
    return df


def summarize_by_children(df: pd.DataFrame) -> dict:
    by_children_binary = df.groupby("children")["has_affair"].agg(["mean", "count"])
    by_children_freq = df.groupby("children")["affairs_freq"].agg(["mean", "median"])

    # Normalize keys to lowercase for robustness
    binary_stats = {
        str(k).lower(): {"prop_any": v["mean"], "n": v["count"]}
        for k, v in by_children_binary.to_dict(orient="index").items()
    }
    freq_stats = {
        str(k).lower(): {"mean_freq": v["mean"], "median_freq": v["median"]}
        for k, v in by_children_freq.to_dict(orient="index").items()
    }

    out = {}
    for key in set(binary_stats) | set(freq_stats):
        out[key] = {**binary_stats.get(key, {}), **freq_stats.get(key, {})}
    return out


def fit_logistic(df: pd.DataFrame):
    df = df.copy()
    df["children_yes"] = df["children"].str.lower().eq("yes").astype(int)
    df["female"] = df["gender"].str.lower().eq("female").astype(int)

    predictors = [
        "children_yes",
        "female",
        "age",
        "years_married",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    X = df[predictors]
    X = sm.add_constant(X, has_constant="add")
    y = df["has_affair"]

    try:
        model = sm.Logit(y, X).fit(disp=False)
        coef = float(model.params["children_yes"])
        pval = float(model.pvalues["children_yes"])
        return {"coef": coef, "pval": pval}
    except Exception:
        return {"coef": np.nan, "pval": np.nan}


def decide_conclusion(group_stats: dict, logit_stats: dict) -> dict:
    yes_stats = group_stats.get("yes", {})
    no_stats = group_stats.get("no", {})

    prop_yes = yes_stats.get("prop_any", np.nan)
    prop_no = no_stats.get("prop_any", np.nan)
    mean_freq_yes = yes_stats.get("mean_freq", np.nan)
    mean_freq_no = no_stats.get("mean_freq", np.nan)

    prop_diff = prop_no - prop_yes  # positive if children group has fewer affairs
    freq_diff = mean_freq_no - mean_freq_yes

    coef = logit_stats.get("coef", np.nan)
    pval = logit_stats.get("pval", np.nan)

    # Determine direction using multiple pieces of evidence
    votes = []
    if not np.isnan(prop_diff):
        votes.append("yes" if prop_diff > 0 else "no")
    if not np.isnan(freq_diff):
        votes.append("yes" if freq_diff > 0 else "no")
    if not np.isnan(coef):
        votes.append("yes" if coef < 0 else "no")

    yes_votes = votes.count("yes")
    no_votes = votes.count("no")

    if yes_votes > no_votes:
        response = "Yes"
    elif no_votes > yes_votes:
        response = "No"
    else:
        # Tie-breaker: rely on logistic sign if available, otherwise default to "No"
        if not np.isnan(coef):
            response = "Yes" if coef < 0 else "No"
        else:
            response = "No"

    # Strength and confidence heuristics
    base_strength = 50
    base_confidence = 50

    magnitude_prop = abs(prop_diff)
    magnitude_freq = abs(freq_diff)

    if magnitude_prop > 0.1:
        base_strength += 15
        base_confidence += 10
    if magnitude_prop > 0.2:
        base_strength += 10
        base_confidence += 10
    if magnitude_freq > 0.5:
        base_strength += 5
    if magnitude_freq > 1.0:
        base_strength += 5

    if not np.isnan(pval):
        if pval < 0.01:
            base_strength += 15
            base_confidence += 20
        elif pval < 0.05:
            base_strength += 10
            base_confidence += 15
        elif pval < 0.1:
            base_strength += 5
            base_confidence += 10
        else:
            base_confidence -= 5

    strength = int(max(0, min(100, base_strength)))
    confidence = int(max(0, min(100, base_confidence)))

    explanation_parts = []

    if not np.isnan(prop_yes) and not np.isnan(prop_no):
        explanation_parts.append(
            f"Among people without children, {prop_no:.1%} reported at least one extramarital affair, "
            f"compared with {prop_yes:.1%} among those with children."
        )
    if not np.isnan(mean_freq_yes) and not np.isnan(mean_freq_no):
        explanation_parts.append(
            f"The average coded frequency of affairs was {mean_freq_no:.2f} without children "
            f"and {mean_freq_yes:.2f} with children."
        )
    if not np.isnan(coef) and not np.isnan(pval):
        direction = "decrease" if coef < 0 else "increase"
        explanation_parts.append(
            f"A logistic regression of any affair on children and demographic controls "
            f"estimated a children coefficient of {coef:.3f} (p-value {pval:.3f}), "
            f"indicating a {direction} in the odds of having an affair for couples with children."
        )

    explanation_parts.append(
        "Combining these patterns, I formed a holistic judgment about whether having children is associated "
        "with lower engagement in extramarital affairs in this sample."
    )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    df = load_data(Path("affairs.csv"))
    group_stats = summarize_by_children(df)
    logit_stats = fit_logistic(df)
    conclusion = decide_conclusion(group_stats, logit_stats)

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

