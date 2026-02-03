import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("crofoot.csv")

    # Relative group size and contest location
    df["size_diff"] = df["n_focal"] - df["n_other"]
    # Positive means focal group is farther from its home center than the other group is from its own
    df["loc_diff"] = df["dist_focal"] - df["dist_other"]

    X = sm.add_constant(df[["size_diff", "loc_diff"]])
    model = sm.Logit(df["win"], X).fit(disp=False)

    # Summary stats to interpret effect directions
    win_rate_by_size = {
        "focal_smaller": df.loc[df["size_diff"] < 0, "win"].mean(),
        "equal": df.loc[df["size_diff"] == 0, "win"].mean(),
        "focal_larger": df.loc[df["size_diff"] > 0, "win"].mean(),
    }
    win_rate_by_location = {
        "focal_closer": df.loc[df["dist_focal"] < df["dist_other"], "win"].mean(),
        "focal_farther": df.loc[df["dist_focal"] > df["dist_other"], "win"].mean(),
    }

    print(model.summary())
    print("\nWin rates by size category:", win_rate_by_size)
    print("Win rates by location category:", win_rate_by_location)


if __name__ == "__main__":
    main()
