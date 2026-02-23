import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size: focal minus other (positive = focal larger).
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Relative location: focal distance from its center minus other distance
    # (negative = contest closer to focal group's home range center).
    df["dist_diff"] = df["feature5"] - df["feature6"]

    # Simple binary indicators for descriptive summaries.
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    df["focal_closer"] = (df["dist_diff"] < 0).astype(int)

    y = df["feature4"]
    X = df[["size_diff", "dist_diff"]]
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False)

    print(model.summary())

    print("\nMean win probability by focal larger vs not:")
    print(df.groupby("focal_larger")["feature4"].mean())

    print("\nMean win probability by focal closer vs not:")
    print(df.groupby("focal_closer")["feature4"].mean())


if __name__ == "__main__":
    main()

