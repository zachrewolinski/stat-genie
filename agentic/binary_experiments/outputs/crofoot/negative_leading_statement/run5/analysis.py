import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("crofoot.csv")

    # Relative predictors
    df["rel_size"] = df["n_focal"] - df["n_other"]
    # Positive means focal closer to its own center than the other group is to theirs
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    X = sm.add_constant(df[["rel_size", "rel_dist"]])
    y = df["win"]

    model = sm.Logit(y, X).fit(disp=False)

    print("Logit model: win ~ rel_size + rel_dist")
    print(model.summary())

    # Simple descriptive checks
    df["size_advantage"] = df["rel_size"] > 0
    df["location_advantage"] = df["rel_dist"] > 0

    win_by_size = df.groupby("size_advantage")["win"].mean()
    win_by_loc = df.groupby("location_advantage")["win"].mean()

    print("\nWin rate by size advantage (rel_size > 0):")
    print(win_by_size)
    print("\nWin rate by location advantage (rel_dist > 0):")
    print(win_by_loc)


if __name__ == "__main__":
    main()
