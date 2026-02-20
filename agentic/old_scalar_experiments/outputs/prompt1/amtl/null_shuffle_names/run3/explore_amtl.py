import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    # Basic sanity checks to infer semantics
    print("\nSummary of numeric columns:")
    print(df.describe())

    # Check relationships between potential count variables
    for a, b in [("genus", "age"), ("genus", "num_amtl"), ("age", "num_amtl")]:
        if a in df.columns and b in df.columns:
            diff = (df[a] > df[b]).sum()
            print(f"Rows where {a} > {b}: {diff}")

    # Value counts for key categoricals
    if "tooth_class" in df.columns:
        print("\nValue counts for tooth_class (actually genus):")
        print(df["tooth_class"].value_counts())
    if "sockets" in df.columns:
        print("\nValue counts for sockets (tooth class within mouth):")
        print(df["sockets"].value_counts())


if __name__ == "__main__":
    main()

