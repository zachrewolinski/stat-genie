import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"].astype(float)

    # Relative group size: focal minus other (positive => focal larger)
    df["rel_size"] = df["feature7"] - df["feature8"]
    df["size_advantage"] = (df["rel_size"] > 0).astype(int)

    # Relative location: other group's distance minus focal's distance
    # Positive values mean the focal group is closer to the center of its home range.
    df["rel_loc"] = df["feature6"] - df["feature5"]
    df["loc_advantage"] = (df["rel_loc"] > 0).astype(int)

    print("Descriptive stats for relative variables:")
    print(df[["rel_size", "rel_loc"]].describe())
    print()

    print("Win rate by size advantage (rows) and outcome (columns):")
    print(pd.crosstab(df["size_advantage"], df["feature4"], normalize="index"))
    print()

    print("Win rate by location advantage (rows) and outcome (columns):")
    print(pd.crosstab(df["loc_advantage"], df["feature4"], normalize="index"))
    print()

    # Logistic regression with continuous predictors
    X_cont = df[["rel_size", "rel_loc"]].astype(float)
    X_cont = sm.add_constant(X_cont)
    logit_cont = sm.Logit(y, X_cont).fit(disp=False)
    print("Logistic regression with continuous predictors:")
    print(logit_cont.summary())
    print()

    # Logistic regression with binary advantage indicators
    X_bin = df[["size_advantage", "loc_advantage"]].astype(float)
    X_bin = sm.add_constant(X_bin)
    logit_bin = sm.Logit(y, X_bin).fit(disp=False)
    print("Logistic regression with binary advantage indicators:")
    print(logit_bin.summary())


if __name__ == "__main__":
    main()

