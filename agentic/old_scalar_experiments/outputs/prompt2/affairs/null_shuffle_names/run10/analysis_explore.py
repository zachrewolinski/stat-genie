import pandas as pd


def main() -> None:
    df = pd.read_csv("affairs.csv")
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    print("\nValue counts for potential children indicator (religiousness):")
    print(df["religiousness"].value_counts(dropna=False))

    print(
        "\nSummary of affair frequency proxy (age column, per metadata description):"
    )
    print(df["age"].describe())
    print("\nUnique values in age:", sorted(df["age"].unique())[:20])

    print("\nShare with any affair (age > 0):", (df["age"] > 0).mean())


if __name__ == "__main__":
    main()
