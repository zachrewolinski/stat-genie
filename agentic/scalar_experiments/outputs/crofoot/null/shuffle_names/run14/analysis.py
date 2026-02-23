import json
from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


def standardize(series: pd.Series, suffix: str) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=series.index, name=f"{series.name}{suffix}")
    return ((series - mean) / std).rename(f"{series.name}{suffix}")


def effect_score(coef: float, p_value: float) -> float:
    if not (np.isfinite(coef) and np.isfinite(p_value)):
        return 0.0
    if p_value <= 0 or p_value > 1:
        return 0.0
    safe_p = max(min(p_value, 1.0), 1e-16)
    # Map p-values: 0.1 -> 0, 0.01 -> ~0.33, 0.001 -> ~0.67, 1e-4 -> 1
    p_component = (-math.log10(safe_p) - 1.0) / 3.0
    p_component = max(0.0, min(1.0, p_component))
    # With standardized predictors, |coef| ~ 1 is a substantial effect
    effect_component = min(1.0, abs(coef) / 1.5)
    return 0.6 * p_component + 0.4 * effect_component


def main() -> None:
    base = Path(__file__).parent

    # Load metadata (used for context in the explanation)
    info_path = base / "info.json"
    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    df = pd.read_csv(base / "crofoot.csv")
    n_rows = len(df)

    # According to info.json descriptions:
    # - m_focal: 1 if focal group won, 0 otherwise (outcome)
    # - f_other: number of individuals in focal group
    # - win: number of individuals in other group
    # - m_other: distance of focal group from the center of its home range
    # - n_focal: distance of other group from the center of its home range
    df = df.copy()
    df["focal_total"] = df["f_other"]
    df["other_total"] = df["win"]
    df["dist_focal_center"] = df["m_other"]
    df["dist_other_center"] = df["n_focal"]

    # Relative group size: log ratio of focal to other group size
    df["rel_group_size"] = np.log(df["focal_total"] / df["other_total"])

    # Contest location: positive when focal group has "home advantage"
    # (other group farther from its center than focal group is from its own)
    df["loc_focal_advantage"] = df["dist_other_center"] - df["dist_focal_center"]

    # Standardize predictors for interpretable effect sizes
    df["rel_group_size_z"] = standardize(df["rel_group_size"], "_z")
    df["loc_focal_advantage_z"] = standardize(df["loc_focal_advantage"], "_z")

    y = df["m_focal"]
    X = df[["rel_group_size_z", "loc_focal_advantage_z"]]
    X = sm.add_constant(X, has_constant="add")

    params = None
    pvalues = None
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
        params = result.params
        pvalues = result.pvalues
    except Exception:
        # If the joint model fails, fall back to separate univariate models
        params = pd.Series({"const": np.nan, "rel_group_size_z": np.nan, "loc_focal_advantage_z": np.nan})
        pvalues = pd.Series({"const": np.nan, "rel_group_size_z": 1.0, "loc_focal_advantage_z": 1.0})
        for col in ["rel_group_size_z", "loc_focal_advantage_z"]:
            Xi = sm.add_constant(df[[col]], has_constant="add")
            try:
                res_i = sm.Logit(y, Xi).fit(disp=False)
                params[col] = res_i.params[col]
                pvalues[col] = res_i.pvalues[col]
            except Exception:
                # Leave defaults for this predictor if even the univariate model fails
                continue

    coef_group = float(params.get("rel_group_size_z", np.nan))
    coef_loc = float(params.get("loc_focal_advantage_z", np.nan))
    p_group = float(pvalues.get("rel_group_size_z", 1.0))
    p_loc = float(pvalues.get("loc_focal_advantage_z", 1.0))

    # Odds ratios for a 1 SD increase in each predictor
    or_group = float(np.exp(coef_group)) if np.isfinite(coef_group) else float("nan")
    or_loc = float(np.exp(coef_loc)) if np.isfinite(coef_loc) else float("nan")

    # Descriptive contrasts: win rates by median split
    def win_rate(mask: pd.Series) -> float:
        subset = df.loc[mask, "m_focal"]
        if subset.empty:
            return float("nan")
        return float(subset.mean())

    median_group = df["rel_group_size_z"].median()
    high_group = df["rel_group_size_z"] >= median_group
    low_group = ~high_group

    win_high_group = win_rate(high_group)
    win_low_group = win_rate(low_group)

    median_loc = df["loc_focal_advantage_z"].median()
    high_loc = df["loc_focal_advantage_z"] >= median_loc
    low_loc = ~high_loc

    win_high_loc = win_rate(high_loc)
    win_low_loc = win_rate(low_loc)

    overall_win_rate = float(df["m_focal"].mean())

    # Translate statistical evidence into a 0–100 Likert response
    group_strength = effect_score(coef_group, p_group)
    loc_strength = effect_score(coef_loc, p_loc)
    combined_strength = (group_strength + loc_strength) / 2.0
    combined_strength = max(0.0, min(1.0, combined_strength))

    response_value = int(round(combined_strength * 100))
    direction = "Yes" if response_value >= 50 else "No"

    def fmt_prob(p: float) -> str:
        if not np.isfinite(p):
            return "NA"
        return f"{p:.2f}"

    def fmt_or(or_value: float) -> str:
        if not np.isfinite(or_value):
            return "NA"
        return f"{or_value:.2f}"

    explanation = (
        f"Research question: {research_question} "
        f"Using {n_rows} recorded intergroup contests, I fit a logistic regression model "
        f"with focal-group victory (m_focal) as the outcome and two standardized predictors: "
        f"relative group size (log ratio of focal to other group size) and contest location "
        f"(how much closer the focal group is to its own home-range center than the opponent is to theirs). "
        f"The overall focal-group win rate is {fmt_prob(overall_win_rate)}. "
        f"For relative group size, the standardized coefficient is {coef_group:.2f} with p-value {p_group:.3f}, "
        f"corresponding to an odds ratio of {fmt_or(or_group)} for a one-standard-deviation increase; "
        f"focal groups that are relatively larger (above the median log size ratio) win about "
        f"{fmt_prob(win_high_group)} of contests versus {fmt_prob(win_low_group)} when they are relatively smaller. "
        f"For contest location, the standardized coefficient is {coef_loc:.2f} with p-value {p_loc:.3f}, "
        f"and an odds ratio of {fmt_or(or_loc)} per one-standard-deviation advantage in being closer to the center of "
        f"their own home range; contests where the focal group has a stronger location advantage (above-median location score) "
        f"have a win rate of {fmt_prob(win_high_loc)} compared to {fmt_prob(win_low_loc)} when their location advantage is weaker. "
        f"Combining the size and location effects, the strength-of-evidence score for an influence of relative group size and "
        f"contest location on victory probability is mapped to {response_value} on a 0–100 Likert scale, "
        f"so the data support a '{direction}' answer: there is "
        f"{'clear' if response_value >= 70 else 'moderate' if response_value >= 50 else 'limited'} "
        f"evidence that both relative group size and contest location meaningfully affect the probability that a focal "
        f"capuchin group wins an intergroup contest."
    )

    conclusion = {"response": response_value, "explanation": explanation}

    conclusion_path = base / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        # Single JSON object, no extra commentary
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

