import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    genus_stats = (
        df.groupby("genus")
        .agg(
            mean_amtl_prop=("amtl_prop", "mean"),
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            n_rows=("amtl_prop", "size"),
        )
        .reset_index()
    )

    print(genus_stats.to_string(index=False))


if __name__ == "__main__":
    main()

