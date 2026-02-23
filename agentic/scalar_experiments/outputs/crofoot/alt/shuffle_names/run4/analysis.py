import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    df = pd.read_csv(data_path)

    # According to info.json descriptions (though column names are shuffled):
    # - m_focal: 1 if focal won, 0 otherwise (outcome)
    # - f_other: number of individuals in focal group
    # - win: number of individuals in other group
    # - m_other: distance (m) of focal group from its home-range center
    # - n_focal: distance (m) of other group from its home-range center
    #
    # We reconstruct key conceptual variables:
    df["focal_win"] = df["m_focal"].astype(int)
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]
    df["dist_focal"] = df["m_other"]
    df["dist_other"] = df["n_focal"]

    # Relative group size: positive when focal group is larger.
    df["rel_group_size"] = df["size_focal"] - df["size_other"]

    # Contest location advantage: positive when focal group is closer to its
    # own home-range center than the other group is to theirs.
    df["loc_advantage"] = df["dist_other"] - df["dist_focal"]

    # Simple descriptive summaries: win rate by relative size category
    def size_category(delta: float) -> str:
        if delta > 0:
            return "focal_larger"
        if delta < 0:
            return "other_larger"
        return "same_size"

    df["size_category"] = df["rel_group_size"].apply(size_category)

    win_by_size = (
        df.groupby("size_category")["focal_win"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "win_rate"})
    )

    # Location advantage categories
    df["focal_home_advantage"] = df["loc_advantage"] > 0
    win_by_location = (
        df.groupby("focal_home_advantage")["focal_win"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "win_rate"})
    )

    # Logistic regression: probability focal group wins as a function of
    # relative group size and location advantage (both continuous).
    X = df[["rel_group_size", "loc_advantage"]].copy()
    # Standardize predictors for easier interpretation of coefficients.
    X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)
    y = df["focal_win"]

    logit_result = sm.Logit(y, X).fit(disp=False)

    # Collect key statistics for downstream interpretation.
    coef = logit_result.params.to_dict()
    pvalues = logit_result.pvalues.to_dict()
    conf_int = logit_result.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]
    conf_int = conf_int.to_dict(orient="index")

    # Approximate odds ratios for a 1 SD increase in each predictor.
    odds_ratios = {name: float(np.exp(val)) for name, val in coef.items()}

    summary = {
        "n_obs": int(logit_result.nobs),
        "win_rate_overall": float(df["focal_win"].mean()),
        "win_by_size": win_by_size.reset_index().to_dict(orient="records"),
        "win_by_location": win_by_location.reset_index().to_dict(orient="records"),
        "logit": {
            "coef": coef,
            "pvalues": pvalues,
            "conf_int": conf_int,
            "odds_ratios": odds_ratios,
        },
    }

    # Write a machine-readable summary for inspection.
    out_path = Path("analysis_summary.json")
    out_path.write_text(json.dumps(summary, indent=2))

    # Also print a concise text summary to stdout for quick review.
    print("Overall focal win rate:", summary["win_rate_overall"])
    print("\nWin rate by relative group size category:")
    for row in summary["win_by_size"]:
        print(
            f"  {row['size_category']}: n={row['count']}, "
            f"win_rate={row['win_rate']:.3f}"
        )

    print("\nWin rate by focal home-range advantage (True = focal closer to center):")
    for row in summary["win_by_location"]:
        print(
            f"  focal_home_advantage={row['focal_home_advantage']}: "
            f"n={row['count']}, win_rate={row['win_rate']:.3f}"
        )

    print("\nLogistic regression coefficients (1 SD-scaled predictors):")
    for name in ["rel_group_size", "loc_advantage"]:
        beta = coef[name]
        p = pvalues[name]
        or_val = odds_ratios[name]
        ci = conf_int[name]
        print(
            f"  {name}: beta={beta:.3f}, OR={or_val:.3f}, "
            f"95% CI=({np.exp(ci['ci_lower']):.3f}, {np.exp(ci['ci_upper']):.3f}), "
            f"p={p:.4f}"
        )


if __name__ == "__main__":
    main()

