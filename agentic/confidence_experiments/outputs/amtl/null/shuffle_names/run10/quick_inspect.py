import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    print("Columns:", list(df.columns))
    print("Head:")
    print(df.head())

    # Check genus vs age relationship to infer which is sockets vs missing
    genus = df["genus"]
    age = df["age"]
    more_missing_than_sockets = (genus > age).sum()
    more_sockets_than_missing = (age > genus).sum()
    equal = (age == genus).sum()
    print("Rows where genus > age:", more_missing_than_sockets)
    print("Rows where age > genus:", more_sockets_than_missing)
    print("Rows where age == genus:", equal)


if __name__ == "__main__":
    main()

