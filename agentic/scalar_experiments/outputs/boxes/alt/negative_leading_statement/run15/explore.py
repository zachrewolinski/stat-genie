import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())

    print("\nValue counts for y:")
    print(df["y"].value_counts().sort_index())

    print("\nAge summary:")
    print(df["age"].describe())

    print("\nCulture counts:")
    print(df["culture"].value_counts().sort_index())

    print("\nY by culture (row-normalized):")
    print(pd.crosstab(df["culture"], df["y"], normalize="index"))

    age_bins = pd.cut(df["age"], bins=[4, 6, 8, 10, 12, 14], right=True, include_lowest=True)
    print("\nY by age group (row-normalized):")
    print(pd.crosstab(age_bins, df["y"], normalize="index"))


if __name__ == "__main__":
    main()

