import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", list(df.columns))
    print("\nHead:")
    print(df.head())

    print("\nValue counts for tooth_class candidate (column 'sockets'):")
    print(df["sockets"].value_counts())

    print("\nValue counts for genus candidate (column 'tooth_class'):")
    print(df["tooth_class"].value_counts())

    print("\nNumeric summary:")
    print(df.describe())


if __name__ == "__main__":
    main()

