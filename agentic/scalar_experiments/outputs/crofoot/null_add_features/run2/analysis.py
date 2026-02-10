import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Basic sanity checks
    print("Rows:", len(df))
    print(df.describe(include="all"))

    # Create key predictors based on the research question
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["focal_larger"] = (df["rel_group_size"] > 0).astype(int)

    # Contest location advantage: focal closer to its own home-range center
    df["focal_home_adv"] = (df["dist_focal"] < df["dist_other"]).astype(int)
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]

    print("\nWin rate by focal larger vs. not:")
    print(df.groupby("focal_larger")["win"].mean())

    print("\nWin rate by focal home-range advantage:")
    print(df.groupby("focal_home_adv")["win"].mean())

    print("\nCorrelation between relative group size and win:")
    print(df[["rel_group_size", "win"]].corr())

    # Logistic regressions
    print("\nLogistic regression: win ~ rel_group_size")
    model_size = smf.logit("win ~ rel_group_size", data=df).fit(disp=False)
    print(model_size.summary())

    print("\nLogistic regression: win ~ focal_home_adv")
    model_home = smf.logit("win ~ focal_home_adv", data=df).fit(disp=False)
    print(model_home.summary())

    print("\nLogistic regression: win ~ rel_group_size + focal_home_adv")
    model_both = smf.logit("win ~ rel_group_size + focal_home_adv", data=df).fit(disp=False)
    print(model_both.summary())

    # Also look at interaction just in case
    print("\nLogistic regression: win ~ rel_group_size * focal_home_adv")
    model_int = smf.logit("win ~ rel_group_size * focal_home_adv", data=df).fit(disp=False)
    print(model_int.summary())


if __name__ == "__main__":
    main()

