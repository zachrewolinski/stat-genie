import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Relative group size: positive when focal group is larger
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["focal_larger"] = (df["rel_size"] > 0).astype(int)

    # Contest location advantage: positive when focal group is closer to its home-range center
    df["rel_dist_center"] = df["dist_other"] - df["dist_focal"]
    df["focal_more_central"] = (df["rel_dist_center"] > 0).astype(int)

    print("N rows:", len(df))
    print("\nOverall win rate (focal wins):", df["win"].mean())

    # Descriptive summaries
    print("\nWin rate by relative group size (binary: focal larger vs not):")
    print(df.groupby("focal_larger")["win"].mean())

    print("\nWin rate by relative location (binary: focal more central vs not):")
    print(df.groupby("focal_more_central")["win"].mean())

    # Logistic regression with continuous predictors
    model_cont = smf.logit(
        formula="win ~ rel_size + rel_dist_center", data=df
    ).fit(disp=False)
    print("\nLogit model with continuous predictors (rel_size, rel_dist_center):")
    print(model_cont.summary())

    # Logistic regression with binary predictors
    model_bin = smf.logit(
        formula="win ~ focal_larger + focal_more_central", data=df
    ).fit(disp=False)
    print("\nLogit model with binary predictors (focal_larger, focal_more_central):")
    print(model_bin.summary())

    # Odds ratios for binary model
    odds_ratios = model_bin.params.apply(lambda x: float(np.exp(x)))
    print("\nApproximate odds ratios (binary model):")
    print(odds_ratios)


if __name__ == "__main__":
    main()
