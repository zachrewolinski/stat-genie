import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won
    y = df["m_focal"]

    # Total group sizes (based on info.json descriptions)
    # f_other: number of individuals in focal group
    # win:    number of individuals in other group
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]

    # Relative size metrics
    df["size_diff"] = df["size_focal"] - df["size_other"]
    df["size_ratio"] = df["size_focal"] / df["size_other"]

    # Location metrics (distances from home range centers)
    # m_other: distance of focal from its home range center
    # n_focal: distance of other from its home range center
    df["dist_focal_home"] = df["m_other"]
    df["dist_other_home"] = df["n_focal"]
    df["home_advantage"] = df["dist_other_home"] - df["dist_focal_home"]

    # Binary indicators for simpler interpretation
    df["larger_focal"] = (df["size_focal"] > df["size_other"]).astype(int)
    df["closer_to_own_home"] = (df["dist_focal_home"] < df["dist_other_home"]).astype(int)

    # Core predictors: relative size and home advantage (continuous)
    X_main = df[["size_diff", "home_advantage"]]
    X_main = sm.add_constant(X_main)

    model_main = sm.Logit(y, X_main)
    result_main = model_main.fit(disp=False)

    print("Main-effects logit coefficients:")
    print(result_main.params)
    print("\nMain-effects p-values:")
    print(result_main.pvalues)

    # Model with interaction between binary size/location indicators
    df["size_loc_interaction"] = df["larger_focal"] * df["closer_to_own_home"]
    X_int = df[["larger_focal", "closer_to_own_home", "size_loc_interaction"]]
    X_int = sm.add_constant(X_int)

    model_int = sm.Logit(y, X_int)
    result_int = model_int.fit(disp=False)

    print("\nBinary+interaction logit coefficients:")
    print(result_int.params)
    print("\nBinary+interaction p-values:")
    print(result_int.pvalues)

    # Also inspect win rates across simple bins for intuition
    print("\nWin rate when focal larger vs not:")
    print(df.groupby("larger_focal")["m_focal"].mean())

    print("\nWin rate when focal closer to its home center vs not:")
    print(df.groupby("closer_to_own_home")["m_focal"].mean())

    print("\nWin rate for combinations of size/location conditions:")
    print(
        df.groupby(["larger_focal", "closer_to_own_home"])["m_focal"].mean().unstack()
    )


if __name__ == "__main__":
    main()
