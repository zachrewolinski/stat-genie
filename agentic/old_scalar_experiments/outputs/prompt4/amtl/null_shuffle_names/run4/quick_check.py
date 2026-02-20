import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic structure
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head())

    # Check ranges to infer semantics
    print("\nSummary for numeric columns:")
    print(df[["genus", "age", "pop", "num_amtl", "stdev_age"]].describe())

    # Check whether one column can be treated as \"missing teeth\"
    # and another as the corresponding number of sockets.
    more_missing_than_sockets = (df["genus"] > df["age"]).sum()
    print(
        f"Rows where genus > age (candidate missing > sockets): "
        f"{more_missing_than_sockets} / {len(df)}"
    )


if __name__ == "__main__":
    main()

