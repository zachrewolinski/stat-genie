import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", list(df.columns))
    print("Head:")
    print(df.head())
    print("\nGenus value counts:")
    print(df["genus"].value_counts(dropna=False))

    print("\nTooth class value counts:")
    print(df["tooth_class"].value_counts(dropna=False))

    # Basic AMTL rate by genus (num_amtl / sockets)
    df = df[df["sockets"] > 0].copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    print("\nAMTL rate by genus (mean, std, n):")
    print(df.groupby("genus")["amtl_rate"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()

