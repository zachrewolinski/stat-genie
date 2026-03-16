import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("Columns:", list(df.columns))
    print("\nBasic info:")
    print(df.info())

    print("\nOutcome y value counts:")
    print(df["y"].value_counts(dropna=False).sort_index())

    print("\nAge summary:")
    print(df["age"].describe())

    print("\nCulture value counts:")
    print(df["culture"].value_counts(dropna=False).sort_index())

    print("\nGender value counts:")
    print(df["gender"].value_counts(dropna=False).sort_index())

    # Social-information use: 1 = non-social (undemonstrated), 2/3 = social
    df["social"] = (df["y"] != 1).astype(int)
    print("\nSocial information use (1=social,0=non-social) value counts:")
    print(df["social"].value_counts(dropna=False).sort_index())

    # Majority preference among those using social information
    social_df = df[df["social"] == 1].copy()
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)
    print("\nMajority choice among social learners (1=majority,0=minority) value counts:")
    print(social_df["majority_choice"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()

