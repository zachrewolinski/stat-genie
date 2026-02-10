import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Map columns to their semantic meanings using info.json descriptions.
    # m_focal: 1 if focal group won, 0 otherwise
    win_focal = df["m_focal"]

    # f_other: number of individuals in focal group
    focal_size = df["f_other"]

    # win: number of individuals in other group
    other_size = df["win"]

    # m_other: distance (m) of focal group from center of its home range
    focal_dist = df["m_other"]

    # n_focal: distance (m) of other group from center of its home range
    other_dist = df["n_focal"]

    # Construct key predictors for the research question.
    df["rel_group_size"] = focal_size - other_size  # positive => focal larger
    df["rel_location"] = other_dist - focal_dist  # positive => focal closer to home

    # Binary indicator for whether the focal group has home-ground advantage.
    df["focal_home_adv"] = (focal_dist < other_dist).astype(int)

    X = df[["rel_group_size", "rel_location", "focal_home_adv"]]
    X = sm.add_constant(X)
    y = win_focal

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression: P(focal win) ~ relative group size + relative location")
    print(result.summary())

    # Report odds ratios for interpretability.
    params = result.params
    conf = result.conf_int()
    odds_ratios = params.apply(lambda b: float(np.exp(b)))

    print("\nOdds ratios (exp(beta)) with 95% CI:")
    for name in params.index:
        or_val = odds_ratios[name]
        ci_low = float(np.exp(conf.loc[name, 0]))
        ci_high = float(np.exp(conf.loc[name, 1]))
        pval = float(result.pvalues[name])
        print(
            f"{name:15s} OR={or_val:6.3f}  "
            f"95% CI=({ci_low:6.3f}, {ci_high:6.3f})  p={pval:.4f}"
        )

    # Simple descriptive checks of win rates.
    print("\nWin rate by relative group size (focal - other):")
    size_cat = pd.cut(
        df["rel_group_size"],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=["much smaller", "slightly smaller", "slightly larger", "much larger"],
    )
    win_rate_by_size = df.groupby(size_cat)["m_focal"].mean()
    print(win_rate_by_size)

    print("\nWin rate by focal home-ground advantage (1=focal closer to home):")
    win_rate_home = df.groupby("focal_home_adv")["m_focal"].mean()
    print(win_rate_home)

    print("\nWin rate by relative location (other_dist - focal_dist) quartiles:")
    loc_quart = pd.qcut(df["rel_location"], q=4, duplicates="drop")
    win_rate_loc = df.groupby(loc_quart)["m_focal"].mean()
    print(win_rate_loc)


if __name__ == "__main__":
    main()
