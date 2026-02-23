import pandas as pd


def main():
    df = pd.read_csv("crofoot.csv")
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["rel_loc"] = df["dist_other"] - df["dist_focal"]

    print("Overall win rate:", df["win"].mean())

    # Relative size categories
    df["size_cat"] = pd.cut(df["rel_size"], [-100, -1, 0, 100], labels=["smaller", "equal", "larger"])
    print("\nWin rate by relative size:")
    print(df.groupby("size_cat")["win"].mean())
    print(df["size_cat"].value_counts())

    # Relative location categories
    df["loc_cat"] = pd.cut(df["rel_loc"], [-10000, -1, 0, 10000], labels=["farther_from_center", "equal", "closer_to_center"])
    print("\nWin rate by relative location:")
    print(df.groupby("loc_cat")["win"].mean())
    print(df["loc_cat"].value_counts())


if __name__ == "__main__":
    main()
