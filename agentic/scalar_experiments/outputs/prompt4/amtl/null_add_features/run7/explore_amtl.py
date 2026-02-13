import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    print("\nGenus value counts:")
    print(df["genus"].value_counts(dropna=False))

    print("\nBasic AMTL vs sockets summary:")
    print(df[["num_amtl", "sockets"]].describe())
    print("Rows with num_amtl > sockets:", (df["num_amtl"] > df["sockets"]).sum())

    df["amtl_any"] = (df["num_amtl"] > 0).astype(int)
    print("\nAMTL presence rate by genus:")
    print(df.groupby("genus")["amtl_any"].mean())


if __name__ == "__main__":
    main()

