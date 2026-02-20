import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Head:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())

    print("\nGenus counts:")
    print(df["genus"].value_counts())

    if "gender" in df.columns:
        print("\nGender unique values:", df["gender"].unique())

    if "prob_male" in df.columns:
        print("\nprob_male summary:")
        print(df["prob_male"].describe())


if __name__ == "__main__":
    main()

