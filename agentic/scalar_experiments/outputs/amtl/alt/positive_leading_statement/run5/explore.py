import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Head:")
    print(df.head(), end="\n\n")

    print("Counts by genus:")
    print(df["genus"].value_counts(), end="\n\n")

    print("Summary of numeric variables:")
    print(df[["num_amtl", "sockets", "age", "stdev_age", "prob_male"]].describe(), end="\n\n")

    print("Raw AMTL rate (num_amtl/sockets) by genus:")
    rate = (df["num_amtl"] / df["sockets"]).groupby(df["genus"]).mean()
    print(rate)


if __name__ == "__main__":
    main()

