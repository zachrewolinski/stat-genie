import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")
    print("Column summaries:\n")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"== {col} ==")
            print(df[col].describe())
            print()


if __name__ == "__main__":
    main()

