import pandas as pd


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic info about columns and head, used interactively while developing.
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())
    print("\nValue counts for 'religiousness' (children yes/no):")
    print(df["religiousness"].value_counts(dropna=False))
    print("\nValue counts for 'age' (affair frequency code):")
    print(df["age"].value_counts().sort_index())


if __name__ == "__main__":
    main()

