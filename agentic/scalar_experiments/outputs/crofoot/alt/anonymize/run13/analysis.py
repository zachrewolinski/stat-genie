import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["win"] = df["feature4"].astype(int)

    # Relative group size: difference and ratio of group sizes (focal - other)
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["size_ratio"] = df["feature7"] / df["feature8"]

    # Contest location: distance of each group from the center of its home range
    # Binary "home advantage" indicator: 1 if focal group is closer to its home-range center
    df["focal_home_adv"] = (df["feature5"] < df["feature6"]).astype(int)
    df["dist_diff"] = df["feature5"] - df["feature6"]

    print("Basic counts")
    print(df[["win", "size_diff", "size_ratio", "focal_home_adv", "dist_diff"]].describe())
    print()

    print("Win rate by relative group size (size_diff sign)")
    df["size_cat"] = np.select(
        [df["size_diff"] < 0, df["size_diff"] == 0, df["size_diff"] > 0],
        ["focal_smaller", "same_size", "focal_larger"],
    )
    print(df.groupby("size_cat")["win"].mean())
    print(df["size_cat"].value_counts())
    print()

    print("Win rate by home-range advantage")
    print(df.groupby("focal_home_adv")["win"].mean())
    print(df["focal_home_adv"].value_counts())
    print()

    # Bivariate logistic regressions
    print("Logit: win ~ size_diff")
    m_size = smf.logit("win ~ size_diff", data=df).fit(disp=False)
    print(m_size.summary())
    print()

    print("Logit: win ~ focal_home_adv")
    m_loc = smf.logit("win ~ focal_home_adv", data=df).fit(disp=False)
    print(m_loc.summary())
    print()

    print("Logit: win ~ size_diff + focal_home_adv")
    m_both = smf.logit("win ~ size_diff + focal_home_adv", data=df).fit(disp=False)
    print(m_both.summary())
    print()

    # Some simple effect summaries
    def prob_from_model(model, **kwargs) -> float:
        params = model.params
        xbeta = params["Intercept"]
        for name, value in kwargs.items():
            xbeta += params.get(name, 0.0) * value
        return float(1.0 / (1.0 + np.exp(-xbeta)))

    print("Estimated win probabilities from joint model:")
    for size_d in (-3, 0, 3):
        for adv in (0, 1):
            p = prob_from_model(m_both, size_diff=size_d, focal_home_adv=adv)
            print(f" size_diff={size_d:+d}, focal_home_adv={adv} -> P(win)={p:.3f}")


if __name__ == "__main__":
    main()
