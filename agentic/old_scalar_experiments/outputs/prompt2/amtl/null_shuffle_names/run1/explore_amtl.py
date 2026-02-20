import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    print("\nGenus (tooth_class column) value counts:")
    print(df["tooth_class"].value_counts())

    print("\nTooth class (sockets column) value counts:")
    print(df["sockets"].value_counts())

    print("\nSex proxy (stdev_age) summary:")
    print(df["stdev_age"].describe())

    print("\nAge at death (pop) summary:")
    print(df["pop"].describe())

    print("\nMissing teeth counts (genus column) summary:")
    print(df["genus"].describe())

    print("\nObservable sockets (age column) summary:")
    print(df["age"].describe())


if __name__ == "__main__":
    main()

