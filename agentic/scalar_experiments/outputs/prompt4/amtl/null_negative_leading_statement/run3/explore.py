import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Head:")
    print(df.head())

    print("\nGenus value counts:")
    print(df["genus"].value_counts())

    print("\nTooth class value counts:")
    print(df["tooth_class"].value_counts())

    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    print("\nSummary of prop_amtl by genus:")
    print(df.groupby("genus")["prop_amtl"].describe())


if __name__ == "__main__":
    main()

