import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Interpret columns based on metadata:
    # m_focal: 1 if focal group won, 0 otherwise (binary outcome)
    # f_other: total individuals in focal group
    # win: total individuals in other group
    # m_other: distance of focal group from its home-range center
    # n_focal: distance of other group from its home-range center

    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]
    df["size_diff"] = df["size_focal"] - df["size_other"]

    df["dist_focal_home"] = df["m_other"]
    df["dist_other_home"] = df["n_focal"]
    df["dist_diff"] = df["dist_focal_home"] - df["dist_other_home"]

    # Descriptive win rates
    overall_win_rate = df["m_focal"].mean()

    larger_mask = df["size_diff"] > 0
    equal_mask = df["size_diff"] == 0
    smaller_mask = df["size_diff"] < 0

    win_larger = df.loc[larger_mask, "m_focal"].mean()
    win_equal = df.loc[equal_mask, "m_focal"].mean()
    win_smaller = df.loc[smaller_mask, "m_focal"].mean()

    # Home-range advantage: focal closer to its own center than other is to its own
    home_adv_mask = df["dist_focal_home"] < df["dist_other_home"]
    away_mask = df["dist_focal_home"] > df["dist_other_home"]

    win_home_adv = df.loc[home_adv_mask, "m_focal"].mean()
    win_away = df.loc[away_mask, "m_focal"].mean()

    # Logistic regression for combined effects
    X = df[["size_diff", "dist_diff"]]
    X = sm.add_constant(X)
    y = df["m_focal"]

    logit_model = sm.Logit(y, X, missing="drop")
    result = logit_model.fit(disp=False)

    print("N contests:", len(df))
    print("Overall focal win rate:", overall_win_rate)
    print()
    print("Win rate when focal larger:", win_larger, " (n =", larger_mask.sum(), ")")
    print("Win rate when sizes equal:", win_equal, " (n =", equal_mask.sum(), ")")
    print("Win rate when focal smaller:", win_smaller, " (n =", smaller_mask.sum(), ")")
    print()
    print("Win rate with home-range advantage (focal closer to home):", win_home_adv, " (n =", home_adv_mask.sum(), ")")
    print("Win rate when focal farther from home than other:", win_away, " (n =", away_mask.sum(), ")")
    print()
    print("Logistic regression results: m_focal ~ size_diff + dist_diff")
    print(result.summary2())


if __name__ == "__main__":
    main()

