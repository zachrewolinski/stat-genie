import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")
    print(df.head())
    print("\nDescribe age:")
    print(df["age"].describe())
    print("\nOutcome counts:")
    print(df["y"].value_counts().sort_index())
    print("\nCultures:", df["culture"].unique())


if __name__ == "__main__":
    main()
