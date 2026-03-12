import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    print("\nDescribe (numeric):")
    print(df.describe())

    print("\nValue counts for tooth_class:")
    print(df["tooth_class"].value_counts())

    print("\nValue counts for sockets:")
    print(df["sockets"].value_counts())

    print("\nMin/max genus vs age:")
    print("genus min,max", df["genus"].min(), df["genus"].max())
    print("age min,max", df["age"].min(), df["age"].max())
    genus_gt_age = df["genus"] > df["age"]
    age_gt_genus = df["age"] > df["genus"]
    print("Any genus>age?", genus_gt_age.any())
    print("Any age>genus?", age_gt_genus.any())
    print("Count genus>age:", genus_gt_age.sum())
    print("Count age>genus:", age_gt_genus.sum())

    over = df[df["genus"] > df["age"]].copy()
    print("\nRows where genus>age (first 10), selected columns:")
    print(over[["sockets", "prob_male", "genus", "age", "pop", "stdev_age", "tooth_class"]].head(10))


if __name__ == "__main__":
    main()
