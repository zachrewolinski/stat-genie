import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    df = df.copy()
    df["prop"] = df["num_amtl"] / df["sockets"]

    print("Shape:", df.shape)
    print("num_amtl min/max:", df["num_amtl"].min(), df["num_amtl"].max())
    print("sockets min/max:", df["sockets"].min(), df["sockets"].max())
    print("prop min/max:", df["prop"].min(), df["prop"].max())
    print("Any num_amtl > sockets:", (df["num_amtl"] > df["sockets"]).any())
    print("Any prop < 0:", (df["prop"] < 0).any())
    print("Any prop > 1:", (df["prop"] > 1).any())
    print("Counts of extreme props:")
    print((df["prop"] == 0).sum(), "rows with prop == 0")
    print((df["prop"] == 1).sum(), "rows with prop == 1")


if __name__ == "__main__":
    main()

