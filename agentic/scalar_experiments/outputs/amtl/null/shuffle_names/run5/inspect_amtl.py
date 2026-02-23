import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Head:")
    print(df.head(), end="\n\n")

    # Basic summaries for numeric columns potentially representing counts
    for col in ["genus", "age"]:
        print(f"Summary for {col}:")
        print(df[col].describe())
        print()

    # Check plausibility of different interpretations of counts
    more_missing_than_sockets_genus_missing = (df["genus"] > df["age"]).sum()
    more_missing_than_sockets_age_missing = (df["age"] > df["genus"]).sum()

    print(
        "Rows where genus>age (if genus=missing, age=sockets):",
        more_missing_than_sockets_genus_missing,
    )
    print(
        "Rows where age>genus (if age=missing, genus=sockets):",
        more_missing_than_sockets_age_missing,
    )


if __name__ == "__main__":
    main()

