import pandas as pd


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]

    grouped = df.groupby("win")[["size_diff", "size_ratio", "dist_diff"]].agg(
        ["mean", "std"]
    )
    print(grouped)


if __name__ == "__main__":
    main()

