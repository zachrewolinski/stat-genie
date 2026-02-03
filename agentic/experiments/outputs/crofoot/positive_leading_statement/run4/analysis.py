import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    df = pd.read_csv("crofoot.csv")

    # Define predictors: relative group size and location advantage
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]  # positive means focal closer to home center
    df["loc_adv_bin"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    print("Rows:", len(df))
    print("Win rate:", df["win"].mean())
    print("Mean win rate when focal closer:", df.loc[df["loc_adv_bin"] == 1, "win"].mean())
    print("Mean win rate when focal farther:", df.loc[df["loc_adv_bin"] == 0, "win"].mean())
    print("Mean win rate when larger group:", df.loc[df["size_diff"] > 0, "win"].mean())
    print("Mean win rate when smaller group:", df.loc[df["size_diff"] < 0, "win"].mean())

    # Logistic regression with continuous location advantage
    X = sm.add_constant(df[["size_diff", "loc_adv"]])
    model = sm.Logit(df["win"], X).fit(disp=False)
    print("\nLogit: win ~ size_diff + loc_adv")
    print(model.summary())

    # Likelihood ratio test vs null
    null = sm.Logit(df["win"], sm.add_constant(pd.DataFrame({"const": [1] * len(df)}))).fit(disp=False)
    lr_stat = 2 * (model.llf - null.llf)
    lr_p = stats.chi2.sf(lr_stat, 2)
    print("LR test vs null: stat=%.3f p=%.3f" % (lr_stat, lr_p))

    # Logistic regression with binary location advantage
    X2 = sm.add_constant(df[["size_diff", "loc_adv_bin"]])
    model2 = sm.Logit(df["win"], X2).fit(disp=False)
    print("\nLogit: win ~ size_diff + loc_adv_bin")
    print(model2.summary())


if __name__ == "__main__":
    main()
