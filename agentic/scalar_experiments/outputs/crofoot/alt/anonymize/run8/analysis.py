import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group wins, 0 otherwise
    df["outcome"] = df["feature4"]

    # Relative group size: focal minus other
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    df["focal_smaller"] = (df["size_diff"] < 0).astype(int)

    # Contest location advantage: 1 if focal group is closer to its home-range center
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    print("Basic description of key variables:")
    print(df[["outcome", "size_diff", "focal_closer"]].describe(), end="\n\n")

    # Win rates by relative group size category
    size_cat = np.select(
        [
            df["size_diff"] > 0,
            df["size_diff"] < 0,
        ],
        ["focal_larger", "focal_smaller"],
        default="equal_size",
    )
    df["size_category"] = size_cat

    print("Win rate by relative group size category (focal vs other):")
    print(df.groupby("size_category")["outcome"].mean(), end="\n\n")

    # Win rates by contest location advantage
    print("Win rate by contest location (focal closer vs not):")
    print(df.groupby("focal_closer")["outcome"].mean(), end="\n\n")

    # Logistic regression: outcome ~ size_diff + focal_closer
    X = df[["size_diff", "focal_closer"]]
    X = sm.add_constant(X)
    y = df["outcome"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    print("Logistic regression: outcome ~ size_diff + focal_closer")
    print(logit_model.summary(), end="\n\n")

    odds_ratios = np.exp(logit_model.params)
    print("Odds ratios:")
    print(odds_ratios, end="\n\n")

    print("p-values:")
    print(logit_model.pvalues, end="\n\n")


if __name__ == "__main__":
    main()

