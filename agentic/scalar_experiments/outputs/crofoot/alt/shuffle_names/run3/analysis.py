import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["m_focal"]

    # Group size variables
    focal_size = df["f_other"]  # number of individuals in focal group
    other_size = df["win"]  # number of individuals in other group
    df["size_diff"] = focal_size - other_size  # positive if focal group is larger
    df["size_ratio"] = focal_size / other_size

    # Location variables: distances from each group's home-range center (meters)
    df["dist_focal_home"] = df["m_other"]
    df["dist_other_home"] = df["n_focal"]
    df["rel_home_advantage"] = df["dist_other_home"] - df["dist_focal_home"]
    df["focal_closer_home"] = (df["dist_focal_home"] < df["dist_other_home"]).astype(int)

    print("Descriptive statistics:")
    print(df[["m_focal", "size_diff", "size_ratio", "dist_focal_home", "dist_other_home", "rel_home_advantage"]].describe())

    print("\nWin rate overall:")
    print(y.mean())

    print("\nWin rate by focal larger / equal / smaller:")
    size_cat = pd.cut(
        df["size_diff"],
        bins=[-100, -0.5, 0.5, 100],
        labels=["focal_smaller", "equal_size", "focal_larger"],
    )
    print(
        pd.crosstab(size_cat, y, normalize="index")
    )  # conditional win probability within each size category

    print("\nWin rate by home-range proximity (focal closer vs not):")
    print(
        pd.crosstab(df["focal_closer_home"], y, normalize="index")
    )  # rows: 0 = other closer/equal, 1 = focal closer

    # Logistic regression models
    print("\nLogistic regression: outcome = win (m_focal) ~ size_diff")
    X_size = sm.add_constant(df["size_diff"])
    model_size = sm.Logit(y, X_size).fit(disp=False)
    print(model_size.summary())

    print("\nLogistic regression: outcome = win (m_focal) ~ focal_closer_home")
    X_loc = sm.add_constant(df["focal_closer_home"])
    model_loc = sm.Logit(y, X_loc).fit(disp=False)
    print(model_loc.summary())

    print("\nLogistic regression: outcome = win (m_focal) ~ size_diff + focal_closer_home")
    X_both = sm.add_constant(df[["size_diff", "focal_closer_home"]])
    model_both = sm.Logit(y, X_both).fit(disp=False)
    print(model_both.summary())

    # Also fit a model using the continuous relative home advantage
    print("\nLogistic regression: outcome = win (m_focal) ~ size_diff + rel_home_advantage")
    X_cont = sm.add_constant(df[["size_diff", "rel_home_advantage"]])
    model_cont = sm.Logit(y, X_cont).fit(disp=False)
    print(model_cont.summary())


if __name__ == "__main__":
    main()

