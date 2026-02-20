import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")
    print(df[["english", "students"]].describe())
    df = df.assign(stratio=df["english"] / df["students"])
    print("\nStudent–teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTop 10 ratios:")
    print(df["stratio"].sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()

