import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["focal_win"] = df["m_focal"]

    # Group sizes (from metadata descriptions)
    df["focal_size"] = df["f_other"]  # number of individuals in focal group
    df["other_size"] = df["win"]      # number of individuals in other group

    # Relative group size (focal minus other) and ratio
    df["rel_group_size"] = df["focal_size"] - df["other_size"]
    df["rel_group_ratio"] = df["focal_size"] / df["other_size"]

    # Contest location: distances to home range centers (in meters)
    # m_other: distance of focal group from its home-range center
    # n_focal: distance of other group from its home-range center
    df["dist_focal_center"] = df["m_other"]
    df["dist_other_center"] = df["n_focal"]

    # Relative location advantage: positive when focal is closer to its center than the opponent
    df["delta_dist"] = df["dist_other_center"] - df["dist_focal_center"]
    df["focal_closer"] = (df["dist_focal_center"] < df["dist_other_center"]).astype(int)

    # Standardize continuous predictors for comparability
    for col in ["rel_group_size", "delta_dist"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        df[f"{col}_z"] = (df[col] - mean) / std

    # Logistic regression: probability that focal group wins
    model = smf.logit(
        "focal_win ~ rel_group_size_z + delta_dist_z",
        data=df,
    ).fit(disp=False)

    print("Logistic regression results: focal_win ~ rel_group_size_z + delta_dist_z")
    print(model.summary())

    # Compute odds ratios and 95% confidence intervals for easier interpretation
    params = model.params
    conf = model.conf_int()
    or_table = pd.DataFrame(
        {
            "odds_ratio": np.exp(params),
            "ci_lower": np.exp(conf[0]),
            "ci_upper": np.exp(conf[1]),
            "p_value": model.pvalues,
        }
    )
    print("\nOdds ratios (per 1 SD increase in predictor):")
    print(or_table)

    # Simple descriptive checks: win rates by relative size and location advantage
    print("\nWin rate when focal larger vs smaller/equal:")
    larger = df["focal_size"] > df["other_size"]
    print(
        df.groupby(larger)["focal_win"].mean().rename(
            {False: "focal_not_larger", True: "focal_larger"}
        )
    )

    print("\nWin rate when focal closer vs not closer to center:")
    print(
        df.groupby("focal_closer")["focal_win"].mean().rename(
            {0: "focal_not_closer", 1: "focal_closer"}
        )
    )


if __name__ == "__main__":
    main()

