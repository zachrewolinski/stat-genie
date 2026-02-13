from pathlib import Path

import pandas as pd


def main():
    df = pd.read_csv(Path("amtl.csv"))

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    invalid_counts = df[df["num_amtl"] > df["sockets"]]
    print("Rows with num_amtl > sockets:", len(invalid_counts))

    nonpositive_sockets = df[df["sockets"] <= 0]
    print("Rows with sockets <= 0:", len(nonpositive_sockets))

    prop = df["num_amtl"] / df["sockets"]
    print("Proportion min/max:", prop.min(), prop.max())
    print("Rows with proportion < 0:", (prop < 0).sum())
    print("Rows with proportion > 1:", (prop > 1).sum())

    missing = df[["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]].isna().sum()
    print("Missing values in key fields:")
    print(missing)


if __name__ == "__main__":
    main()

