import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Rename for clarity
    df = df.rename(
        columns={
            "feature4": "win",  # 1 if focal won
            "feature5": "focal_dist",
            "feature6": "other_dist",
            "feature7": "focal_size",
            "feature8": "other_size",
        }
    )

    # Relative group size: focal minus other (positive = focal larger)
    df["size_diff"] = df["focal_size"] - df["other_size"]
    df["size_advantage"] = (df["size_diff"] > 0).astype(int)

    # Contest location advantage: other_dist - focal_dist
    # Positive values mean focal group is closer to its home-range center
    df["loc_adv"] = df["other_dist"] - df["focal_dist"]
    df["home_advantage"] = (df["loc_adv"] > 0).astype(int)

    print("Number of contests:", len(df))
    print()

    # Descriptive win rates by size advantage
    print("Win rate by size advantage (focal larger vs not):")
    size_win_table = pd.crosstab(df["size_advantage"], df["win"], normalize="index")
    print(size_win_table)
    print()

    # Descriptive win rates by home-range advantage
    print("Win rate by home-range advantage (focal closer to its center vs not):")
    home_win_table = pd.crosstab(df["home_advantage"], df["win"], normalize="index")
    print(home_win_table)
    print()

    # Logistic regression: outcome on relative group size and location advantage
    y = df["win"]
    X = df[["size_diff", "loc_adv"]]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=False)
    print("Logistic regression: win ~ size_diff + loc_adv")
    print(model.summary())
    print()
    print("Odds ratios:")
    print(np.exp(model.params))
    print()

    # Univariate models for robustness
    X_size = sm.add_constant(df[["size_diff"]])
    model_size = sm.Logit(y, X_size).fit(disp=False)
    print("Univariate logistic regression: win ~ size_diff")
    print(model_size.summary())
    print("Odds ratio (size_diff):", np.exp(model_size.params["size_diff"]))
    print()

    X_loc = sm.add_constant(df[["loc_adv"]])
    model_loc = sm.Logit(y, X_loc).fit(disp=False)
    print("Univariate logistic regression: win ~ loc_adv")
    print(model_loc.summary())
    print("Odds ratio (loc_adv):", np.exp(model_loc.params["loc_adv"]))


if __name__ == "__main__":
    main()

