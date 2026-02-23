import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Basic derived variables
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["loc_diff"] = df["dist_focal"] - df["dist_other"]

    print("Basic description:")
    print(df[["win", "size_diff", "size_ratio", "loc_diff"]].describe())
    print()

    # Win rates by relative size
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    df["focal_closer_home"] = (df["loc_diff"] < 0).astype(int)

    def win_rate(group_col: str) -> None:
        grouped = df.groupby(group_col)["win"].agg(["mean", "count"])
        print(f"Win rate by {group_col}:")
        print(grouped)
        print()

    win_rate("focal_larger")
    win_rate("focal_closer_home")

    # Logistic regression: win ~ size_diff + loc_diff
    X = df[["size_diff", "loc_diff"]]
    X = sm.add_constant(X)
    y = df["win"]

    logit_model = sm.Logit(y, X).fit(disp=False)
    print("Logit: win ~ size_diff + loc_diff")
    print(logit_model.summary())
    print()

    # Alternative specification with ratios
    X2 = df[["size_ratio", "loc_diff"]]
    X2 = sm.add_constant(X2)
    logit_model2 = sm.Logit(y, X2).fit(disp=False)
    print("Logit: win ~ size_ratio + loc_diff")
    print(logit_model2.summary())


if __name__ == "__main__":
    main()

