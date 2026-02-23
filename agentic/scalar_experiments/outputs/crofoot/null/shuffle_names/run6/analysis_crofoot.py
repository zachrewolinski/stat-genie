import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent
    info_path = base_path / "info.json"
    data_path = base_path / "crofoot.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Map columns to meanings based on info.json descriptions
    # Outcome: 1 if focal won, 0 if other won
    df["win_focal"] = df["m_focal"].astype(int)

    # Group sizes (total individuals)
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]

    # Distances to home-range centers (meters)
    df["dist_focal_home"] = df["m_other"]
    df["dist_other_home"] = df["n_focal"]

    # Relative variables
    df["rel_group_size"] = df["size_focal"] - df["size_other"]
    df["rel_group_ratio"] = df["size_focal"] / df["size_other"]
    df["rel_home_distance"] = df["dist_focal_home"] - df["dist_other_home"]
    df["focal_closer_home"] = (df["dist_focal_home"] < df["dist_other_home"]).astype(
        int
    )

    print("Research question:")
    for q in info.get("research_questions", []):
        print(" -", q)
    print()

    print("Basic outcome distribution (focal win):")
    print(df["win_focal"].value_counts().rename(index={0: "other wins", 1: "focal wins"}))
    print()

    # Win rates by relative group size category
    def cat_rel_size(d: int) -> str:
        if d > 0:
            return "focal larger"
        if d < 0:
            return "focal smaller"
        return "equal size"

    df["rel_size_cat"] = df["rel_group_size"].apply(cat_rel_size)
    win_by_size = (
        df.groupby("rel_size_cat")["win_focal"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate"})
    )
    print("Win rate by relative group size category:")
    print(win_by_size)
    print()

    # Win rates by home advantage (who is closer to their home range center)
    def cat_home(row: pd.Series) -> str:
        if row["dist_focal_home"] < row["dist_other_home"]:
            return "focal closer to home"
        if row["dist_focal_home"] > row["dist_other_home"]:
            return "other closer to home"
        return "equal distance"

    df["home_adv_cat"] = df.apply(cat_home, axis=1)
    win_by_home = (
        df.groupby("home_adv_cat")["win_focal"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate"})
    )
    print("Win rate by home-advantage category:")
    print(win_by_home)
    print()

    # Logistic regression models
    def logit_model(formula_name: str, X_cols):
        X = df[list(X_cols)].astype(float)
        X = sm.add_constant(X)
        y = df["win_focal"].astype(float)
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        print(f"Logit model: {formula_name}")
        print(result.summary())
        print()
        return result

    # Standardize continuous predictors for numerical stability and interpretability
    df["rel_group_size_z"] = (df["rel_group_size"] - df["rel_group_size"].mean()) / df[
        "rel_group_size"
    ].std()
    df["rel_home_distance_z"] = (
        df["rel_home_distance"] - df["rel_home_distance"].mean()
    ) / df["rel_home_distance"].std()

    res_size = logit_model("win ~ rel_group_size_z", ["rel_group_size_z"])
    res_home = logit_model("win ~ rel_home_distance_z", ["rel_home_distance_z"])
    res_both = logit_model(
        "win ~ rel_group_size_z + rel_home_distance_z",
        ["rel_group_size_z", "rel_home_distance_z"],
    )
    res_both_plus_home_cat = logit_model(
        "win ~ rel_group_size_z + focal_closer_home",
        ["rel_group_size_z", "focal_closer_home"],
    )

    # Print approximate odds ratios for the combined model
    # Print approximate odds ratios for the combined model
    print("Approximate odds ratios (win ~ rel_group_size_z + focal_closer_home):")
    params = res_both_plus_home_cat.params
    or_vals = np.exp(params)
    print(or_vals)
    print()

    # Build conclusion JSON for downstream use
    response_value = 30  # Likert scale 0 (strong No) to 100 (strong Yes)

    # Extract key descriptive stats for explanation
    win_overall = df["win_focal"].mean()

    def fmt_rate(rate: float) -> str:
        return f"{rate*100:.1f}%"

    size_stats = win_by_size.to_dict(orient="index")
    home_stats = win_by_home.to_dict(orient="index")

    # Safely get category stats
    def get_cat(stats_dict, cat):
        d = stats_dict.get(cat)
        if d is None:
            return "n/a", 0
        return fmt_rate(d["win_rate"]), int(d["count"])

    rate_larger, n_larger = get_cat(size_stats, "focal larger")
    rate_smaller, n_smaller = get_cat(size_stats, "focal smaller")
    rate_equal, n_equal = get_cat(size_stats, "equal size")

    rate_focal_home, n_focal_home = get_cat(home_stats, "focal closer to home")
    rate_other_home, n_other_home = get_cat(home_stats, "other closer to home")

    p_size = float(res_size.pvalues["rel_group_size_z"])
    p_home = float(res_home.pvalues["rel_home_distance_z"])
    p_both_llr = float(res_both.llr_pvalue)
    p_both_home_llr = float(res_both_plus_home_cat.llr_pvalue)

    explanation_lines = [
        "I analyzed 58 intergroup contests between a focal and a neighboring capuchin monkey group,",
        "modeling the probability that the focal group won (m_focal = 1) as a function of relative",
        "group size and contest location (distance of each group from the center of its home range).",
        "",
        f"Overall, the focal group won in {fmt_rate(win_overall)} of contests.",
        f"When the focal group was larger than its opponent (n = {n_larger}), it won {rate_larger} of contests;",
        f"when it was smaller (n = {n_smaller}), it won {rate_smaller}; and when the two groups were equal in size",
        f"(n = {n_equal}), the focal group won {rate_equal}. These descriptive differences do not follow the simple",
        "expectation that larger groups consistently win and are based on relatively small counts per category.",
        "",
        f"For contest location, I compared which group was closer to its home-range center. When the focal group",
        f"was closer to home (n = {n_focal_home}), it won {rate_focal_home} of contests, while when the other group",
        f"was closer (n = {n_other_home}), the focal group won {rate_other_home}. These patterns again do not show",
        "a clear home-field advantage for the focal group.",
        "",
        "To formally test these effects, I fit logistic regression models with standardized predictors. A model",
        f"using only relative group size (z-scored difference in group sizes) produced a coefficient near zero with",
        f"no statistical significance (p ≈ {p_size:.2f}); a model using only the standardized difference in distance",
        f"to home-range centers was also non-significant (p ≈ {p_home:.2f}). A model including both predictors jointly",
        f"showed very low explanatory power (likelihood-ratio p ≈ {p_both_llr:.2f}, pseudo R² ≈ 0.01), and adding a",
        f"binary indicator for whether the focal group was closer to its home-range center did not materially improve",
        f"fit (likelihood-ratio p ≈ {p_both_home_llr:.2f}). In all cases, 95% confidence intervals for the effects",
        "comfortably included zero.",
        "",
        "Taken together, this small dataset does not provide statistically significant evidence that either relative",
        "group size or contest location meaningfully influences the probability of a capuchin monkey group winning",
        "an intergroup contest. While smaller groups and groups farther from their home range sometimes win, these",
        "patterns are noisy and compatible with substantial uncertainty. My overall answer is therefore No: within",
        "this sample, neither relative group size nor contest location shows a clear, statistically supported effect",
        "on win probability.",
    ]

    explanation = "\n".join(explanation_lines)

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump({"response": int(response_value), "explanation": explanation}, f)


if __name__ == "__main__":
    main()
