import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    print("Columns:", df.columns.tolist())
    print()
    print("Head:")
    print(df.head())
    print()
    print("dtypes:")
    print(df.dtypes)
    print()
    print("Unique values for 'tooth_class' (genus):", df["tooth_class"].unique())
    print("Unique values for 'sockets' (tooth class):", df["sockets"].unique())
    print()
    print("Summary of numeric columns:")
    print(df.describe())


if __name__ == "__main__":
    main()

