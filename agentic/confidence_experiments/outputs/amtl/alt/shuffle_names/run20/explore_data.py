import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    print("Head:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nUnique values:")
    for col in df.columns:
        uniques = df[col].unique()
        print(f"\nColumn: {col}")
        print(f"n_unique={len(uniques)}")
        print(f"sample_unique={uniques[:10]}")


if __name__ == "__main__":
    main()

