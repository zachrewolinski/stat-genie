import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal won, 0 if other won
    df["win_focal"] = df["m_focal"].astype(int)

    # Group sizes: description indicates f_other = size of focal group, win = size of other group
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]
    df["size_diff"] = df["size_focal"] - df["size_other"]

    # Distances: description indicates m_other = focal distance, n_focal = other distance
    df["dist_focal_home"] = df["m_other"]
    df["dist_other_home"] = df["n_focal"]

    # Relative contest location: negative -> closer to focal home range centre, positive -> closer to other group
    df["rel_dist"] = df["dist_focal_home"] - df["dist_other_home"]

    # Binary indicator: 1 if contest is closer to focal group's home range centre
    df["focal_home_adv"] = (df["rel_dist"] < 0).astype(int)

    print("Basic counts")
    print(df[["win_focal"]].value_counts(normalize=True))
    print()

    print("Win rate by relative group size sign")
    df["size_sign"] = np.sign(df["size_diff"]).replace({-1: "focal_smaller", 0: "equal", 1: "focal_larger"})
    print(df.groupby("size_sign")["win_focal"].mean())
    print()

    print("Win rate by home advantage")
    print(df.groupby("focal_home_adv")["win_focal"].mean())
    print()

    # Logistic regression: win_focal ~ size_diff + focal_home_adv
    X = df[["size_diff", "focal_home_adv"]]
    X = sm.add_constant(X)
    y = df["win_focal"]

    logit_model = sm.Logit(y, X).fit(disp=False)
    print("Logistic regression: win_focal ~ size_diff + focal_home_adv")
    print(logit_model.summary())


if __name__ == "__main__":
    main()

