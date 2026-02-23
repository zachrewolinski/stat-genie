import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Map columns to their semantic meanings based on info.json descriptions.
    win_focal = df["m_focal"]  # 1 if focal won, 0 if other won

    size_focal = df["f_other"]  # number of individuals in focal group
    size_other = df["win"]  # number of individuals in other group

    # Distances from each group's home range center (meters).
    dist_focal_center = df["m_other"]  # focal group's distance
    dist_other_center = df["n_focal"]  # other group's distance

    df = df.assign(
        win_focal=win_focal,
        size_focal=size_focal,
        size_other=size_other,
        rel_size=size_focal - size_other,
        dist_focal_center=dist_focal_center,
        dist_other_center=dist_other_center,
        rel_dist=dist_focal_center - dist_other_center,  # <0: focal closer to home center
        focal_closer=(dist_focal_center < dist_other_center).astype(int),
        focal_larger=(size_focal > size_other).astype(int),
        dyad=df["dyad"],
    )

    print("Basic dataset info")
    print("-------------------")
    print(df[["win_focal", "size_focal", "size_other", "rel_size", "dist_focal_center", "dist_other_center", "rel_dist"]].describe())
    print()

    overall_win_rate = df["win_focal"].mean()
    print(f"Overall focal win rate: {overall_win_rate:.3f} (n={len(df)})")
    print()

    # Descriptive win rates by relative size.
    print("Win rate by relative group size (focal vs other)")
    print("------------------------------------------------")
    size_cats = pd.cut(
        df["rel_size"],
        bins=[-100, -1, 0, 1, 100],
        labels=["focal_smaller", "equal", "focal_slightly_larger", "focal_much_larger"],
        include_lowest=True,
    )
    print(
        df.groupby(size_cats, observed=True)["win_focal"].agg(["mean", "count"])
    )
    print()

    # Descriptive win rates by relative distance (home advantage).
    print("Win rate by which group is closer to its home range center")
    print("-----------------------------------------------------------")
    dist_group = pd.Series("equal", index=df.index)
    dist_group[df["rel_dist"] < 0] = "focal_closer"
    dist_group[df["rel_dist"] > 0] = "other_closer"
    print(
        df.groupby(dist_group)["win_focal"].agg(["mean", "count"])
    )
    print()

    # Combined descriptive: size advantage and home advantage.
    print("Win rate by size advantage and home advantage")
    print("--------------------------------------------")
    combo = pd.crosstab(
        df["focal_larger"].map({0: "focal_not_larger", 1: "focal_larger"}),
        df["focal_closer"].map({0: "focal_not_closer", 1: "focal_closer"}),
        values=df["win_focal"],
        aggfunc=["mean", "count"],
    )
    print(combo)
    print()

    # Logistic regression: probability of focal win as a function of relative size and relative distance.
    print("Logistic regression: win_focal ~ rel_size + rel_dist")
    print("----------------------------------------------------")
    X = df[["rel_size", "rel_dist"]].astype(float)
    X = sm.add_constant(X)
    y = df["win_focal"]

    logit_model = sm.Logit(y, X)
    logit_res = logit_model.fit(disp=False)
    print(logit_res.summary())
    print()

    # Also fit models with each predictor separately to see their individual contributions.
    print("Logistic regression: win_focal ~ rel_size")
    print("-----------------------------------------")
    X_size = sm.add_constant(df[["rel_size"]].astype(float))
    logit_size = sm.Logit(y, X_size).fit(disp=False)
    print(logit_size.summary())
    print()

    print("Logistic regression: win_focal ~ rel_dist")
    print("-----------------------------------------")
    X_dist = sm.add_constant(df[["rel_dist"]].astype(float))
    logit_dist = sm.Logit(y, X_dist).fit(disp=False)
    print(logit_dist.summary())
    print()

    # Logistic regression using binary indicators for size and home advantage.
    print("Logistic regression: win_focal ~ focal_larger + focal_closer")
    print("------------------------------------------------------------")
    X_bin = sm.add_constant(df[["focal_larger", "focal_closer"]].astype(float))
    logit_bin = sm.Logit(y, X_bin).fit(disp=False)
    print(logit_bin.summary())
    print()


if __name__ == "__main__":
    main()
