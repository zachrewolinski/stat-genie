import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # According to metadata in info.json:
    # m_focal: 1 if focal won, 0 if other won (binary outcome)
    # f_other: number of individuals in focal group
    # win: number of individuals in other group
    # m_other: distance (m) of focal group from center of its home range
    # n_focal: distance (m) of other group from center of its home range

    # Sanity check basic structure
    print("Head:")
    print(df.head())
    print("\nDescribe:")
    print(df.describe())

    # Construct predictors
    # Relative group size: focal size minus other size and ratio
    size_focal = df["f_other"]
    size_other = df["win"]

    df["size_diff"] = size_focal - size_other
    df["size_ratio"] = size_focal / size_other

    # Home-range proximity: smaller distance to own center = home advantage
    dist_focal = df["m_other"]  # distance of focal group from its home center
    dist_other = df["n_focal"]  # distance of other group from its home center

    df["home_adv"] = dist_other - dist_focal  # positive if other further from home than focal

    # Outcome
    y = df["m_focal"]

    # Design matrix with intercept
    X = df[["size_diff", "size_ratio", "home_adv"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("\nLogit summary:")
    print(result.summary())

    # Extract key statistics
    params = result.params
    pvalues = result.pvalues

    print("\nParams:")
    print(params)
    print("\nP-values:")
    print(pvalues)

    # Simple descriptive checks: win rate when focal larger, equal, smaller
    larger = df[df["size_diff"] > 0]["m_focal"]
    equal_ = df[df["size_diff"] == 0]["m_focal"]
    smaller = df[df["size_diff"] < 0]["m_focal"]

    print("\nWin rate when focal larger:", larger.mean(), "n=", len(larger))
    print("Win rate when equal size:", equal_.mean(), "n=", len(equal_))
    print("Win rate when focal smaller:", smaller.mean(), "n=", len(smaller))

    # Home advantage: focal closer to home vs other closer
    focal_home = df[dist_focal < dist_other]["m_focal"]
    other_home = df[dist_focal > dist_other]["m_focal"]
    neutral = df[dist_focal == dist_other]["m_focal"]

    print("\nWin rate when focal closer to home:", focal_home.mean(), "n=", len(focal_home))
    print("Win rate when other closer to home:", other_home.mean(), "n=", len(other_home))
    print("Win rate when equal distance:", neutral.mean(), "n=", len(neutral))


if __name__ == "__main__":
    main()
