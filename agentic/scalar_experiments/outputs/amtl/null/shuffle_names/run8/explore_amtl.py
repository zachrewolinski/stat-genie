import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    print("\nValue counts for genus label (tooth_class column):")
    print(df["tooth_class"].value_counts())

    print("\nValue counts for tooth class (sockets column):")
    print(df["sockets"].value_counts())

    print("\nSummary of numeric columns:")
    print(df[["genus", "age", "pop", "num_amtl", "stdev_age"]].describe())

    # Check that total teeth (present + missing) is always >= missing and positive
    total_teeth = df["genus"] + df["age"]
    invalid = (total_teeth <= 0) | (df["genus"] < 0)
    print("\nAny invalid total/missing combinations:", invalid.any())
    print("Min total_teeth:", total_teeth.min(), "Max total_teeth:", total_teeth.max())


if __name__ == "__main__":
    main()

