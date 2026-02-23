import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Map columns to more interpretable names based on info.json descriptions.
    df = df.assign(
        size_focal=df["f_other"],  # number of individuals in focal group
        size_other=df["win"],  # number of individuals in other group
        dist_focal_hr=df["m_other"],  # distance of focal group from its home-range center (m)
        dist_other_hr=df["n_focal"],  # distance of other group from its home-range center (m)
    )

    # Relative group size metrics
    df["size_diff"] = df["size_focal"] - df["size_other"]
    df["size_ratio"] = df["size_focal"] / df["size_other"]
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)

    # Contest location metrics (home-range advantage)
    df["home_advantage_diff"] = df["dist_other_hr"] - df["dist_focal_hr"]
    df["focal_closer_home"] = (df["dist_focal_hr"] < df["dist_other_hr"]).astype(int)

    # Outcome: 1 if focal group won, 0 otherwise
    df["win_focal"] = df["m_focal"]

    print("=== Basic descriptives ===")
    print("N rows:", len(df))
    print()

    # Win rate by relative group size
    print("Win rate by whether focal group is larger:")
    print(
        df.groupby("focal_larger")["win_focal"].agg(["mean", "count"])
        .rename(index={0: "focal_not_larger", 1: "focal_larger"})
    )
    print()

    # Win rate by home-range advantage
    print("Win rate by whether focal group is closer to its home range center:")
    print(
        df.groupby("focal_closer_home")["win_focal"].agg(["mean", "count"])
        .rename(index={0: "focal_not_closer", 1: "focal_closer"})
    )
    print()

    # Logistic regression models
    print("=== Logistic regression: continuous predictors ===")
    model_cont = smf.logit(
        "win_focal ~ size_diff + home_advantage_diff", data=df
    ).fit(disp=False)
    print(model_cont.summary())
    print()

    print("=== Logistic regression: binary advantage indicators ===")
    model_bin = smf.logit(
        "win_focal ~ focal_larger + focal_closer_home", data=df
    ).fit(disp=False)
    print(model_bin.summary())
    print()

    print("Odds ratios (binary model):")
    or_bin = np.exp(model_bin.params)
    print(or_bin)


if __name__ == "__main__":
    main()

