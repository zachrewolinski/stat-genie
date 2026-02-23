import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size: positive values mean the focal group is larger.
    df["rel_group_size"] = df["n_focal"] - df["n_other"]

    # Relative location: positive values mean the focal group is farther
    # from its home range center than the other group (disadvantage).
    df["rel_dist"] = df["dist_focal"] - df["dist_other"]

    # Simple categorical indicators for descriptive summaries.
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)
    df["focal_closer_home"] = (df["rel_dist"] < 0).astype(int)

    y = df["win"]
    X = df[["rel_group_size", "rel_dist"]]
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X).fit(disp=False)

    params = model.params
    pvalues = model.pvalues
    odds_ratios = np.exp(params)

    print("Logistic regression: win ~ rel_group_size + rel_dist")
    print(model.summary2())
    print("\nOdds ratios:")
    for name, value in odds_ratios.items():
        print(f"  {name}: {value:.3f}")

    print("\nP-values:")
    for name, value in pvalues.items():
        print(f"  {name}: {value:.4f}")

    # Descriptive win rates by relative group size and location.
    win_rate_larger = df.loc[df["focal_larger"] == 1, "win"].mean()
    win_rate_smaller_or_equal = df.loc[df["focal_larger"] == 0, "win"].mean()

    win_rate_home = df.loc[df["focal_closer_home"] == 1, "win"].mean()
    win_rate_away_or_neutral = df.loc[df["focal_closer_home"] == 0, "win"].mean()

    print("\nDescriptive win rates:")
    print(f"  Focal larger group win rate: {win_rate_larger:.3f}")
    print(f"  Focal smaller/equal group win rate: {win_rate_smaller_or_equal:.3f}")
    print(f"  Focal closer to home win rate: {win_rate_home:.3f}")
    print(f"  Focal farther/equal distance win rate: {win_rate_away_or_neutral:.3f}")


if __name__ == "__main__":
    main()

