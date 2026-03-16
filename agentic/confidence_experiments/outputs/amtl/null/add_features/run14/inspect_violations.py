import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    mask = df["num_amtl"] > df["sockets"]
    print(f"Rows with num_amtl > sockets: {mask.sum()}")
    if mask.any():
        print(df.loc[mask].to_string(index=False))


if __name__ == "__main__":
    main()

