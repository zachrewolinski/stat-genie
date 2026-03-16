import pandas as pd


def main() -> None:
    df = pd.read_csv("affairs.csv")

    print("Head:")
    print(df.head())
    print("\nInfo:")
    print(df.info())

    print("\nDescribe (numeric):")
    print(df.describe())

    if "religiousness" in df.columns:
        print("\nValue counts for religiousness (children indicator per metadata):")
        print(df["religiousness"].value_counts())

    if "age" in df.columns:
        print("\nUnique values in age (affair frequency candidate):")
        print(sorted(df["age"].unique()))


if __name__ == "__main__":
    main()

