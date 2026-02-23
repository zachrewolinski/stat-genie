import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Derived predictors capturing relative group size and contest location
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_center_dist"] = df["dist_other"] - df["dist_focal"]
    df["focal_home_adv"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    print("Basic description")
    print(df[["win", "rel_group_size", "rel_center_dist", "focal_home_adv"]].describe())
    print("\nProportion of wins by relative group size sign:")
    sign = df["rel_group_size"].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))
    tmp = df.groupby(sign)["win"].agg(["mean", "count"])
    print(tmp)

    print("\nProportion of wins by home advantage:")
    print(df.groupby("focal_home_adv")["win"].agg(["mean", "count"]))

    # Logistic regressions
    print("\nModel A: win ~ rel_group_size")
    model_a = smf.glm(
        formula="win ~ rel_group_size",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_a.summary())

    print("\nModel B: win ~ rel_center_dist")
    model_b = smf.glm(
        formula="win ~ rel_center_dist",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_b.summary())

    print("\nModel C: win ~ rel_group_size + rel_center_dist")
    model_c = smf.glm(
        formula="win ~ rel_group_size + rel_center_dist",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_c.summary())

    print("\nModel D: win ~ rel_group_size + focal_home_adv")
    model_d = smf.glm(
        formula="win ~ rel_group_size + focal_home_adv",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_d.summary())


if __name__ == "__main__":
    main()

