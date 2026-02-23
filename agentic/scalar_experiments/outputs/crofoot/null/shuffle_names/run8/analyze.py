import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(".")

    info_path = base_path / "info.json"
    data_path = base_path / "crofoot.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    print("Research question:")
    for q in info.get("research_questions", []):
        print(" -", q)
    print()

    df = pd.read_csv(data_path)
    print("Columns:", list(df.columns))
    print("Number of rows:", len(df))

    # Map columns to their semantic meaning using descriptions from info.json.
    # Note: column *names* are shuffled; we rely on descriptions here.
    # Outcome: 1 if focal group won, 0 otherwise.
    win_focal = df["m_focal"]
    assert set(win_focal.unique()) <= {0, 1}, "m_focal should be a binary win indicator."

    # Group sizes: total number of individuals in focal vs other group.
    size_focal = df["f_other"]  # "Number of individuals in focal group"
    size_other = df["win"]  # "Number of individuals in other group"

    # Distances from each group's home-range center (contest location proxy).
    dist_focal_center = df["m_other"]  # "Distance of focal group from center of its home range"
    dist_other_center = df["n_focal"]  # "Distance of other group from center of its home range"

    # Derived predictors capturing *relative* size and location advantage.
    df["log_size_ratio"] = np.log(size_focal / size_other)
    df["size_diff"] = size_focal - size_other

    # Positive dist_diff => other group is farther from its center than focal is from its own
    # (i.e., focal has a location advantage).
    df["dist_diff"] = dist_other_center - dist_focal_center

    # Categorical home-advantage indicator.
    def which_home(row) -> str:
        if row["dist_focal_center"] < row["dist_other_center"]:
            return "focal_home"
        elif row["dist_focal_center"] > row["dist_other_center"]:
            return "other_home"
        return "neutral"

    df["dist_focal_center"] = dist_focal_center
    df["dist_other_center"] = dist_other_center
    df["home_adv"] = df.apply(which_home, axis=1)

    print("\nBasic summaries:")
    print(df[["log_size_ratio", "size_diff", "dist_diff"]].describe())

    print("\nWin rate overall:", win_focal.mean())
    print("Win rate when focal larger:", win_focal[df["log_size_ratio"] > 0].mean())
    print("Win rate when focal smaller:", win_focal[df["log_size_ratio"] < 0].mean())

    print("\nWin rate by home advantage:")
    print(df.groupby("home_adv")["m_focal"].mean())

    # Logistic regression: P(win) ~ relative size + relative location.
    y = win_focal
    X = df[["log_size_ratio", "dist_diff"]]
    X = sm.add_constant(X)

    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    print("\nLogistic regression (standard errors):")
    print(res.summary())

    # Cluster-robust SEs by dyad (group pair) to account for repeated contests.
    try:
        res_cluster = logit.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["dyad"]})
        print("\nLogistic regression with dyad-clustered SEs:")
        print(res_cluster.summary())
    except Exception as e:  # pragma: no cover - robustness only
        print("\nClustered SE fit failed:", repr(e))


if __name__ == "__main__":
    main()

