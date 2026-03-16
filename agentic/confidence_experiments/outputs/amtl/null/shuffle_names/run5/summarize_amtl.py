import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")
    df = df.rename(
        columns={
            "genus": "missing_count",
            "age": "socket_count",
            "tooth_class": "genus_taxon",
        }
    )

    df = df[
        (df["socket_count"] > 0)
        & (df["missing_count"] >= 0)
        & df["socket_count"].notna()
        & df["missing_count"].notna()
        & (df["missing_count"] <= df["socket_count"])
    ].copy()

    df["missing_rate"] = df["missing_count"] / df["socket_count"]

    genus_summary = (
        df.groupby("genus_taxon")
        .agg(
            total_missing=("missing_count", "sum"),
            total_sockets=("socket_count", "sum"),
            mean_rate=("missing_rate", "mean"),
        )
        .assign(overall_rate=lambda x: x["total_missing"] / x["total_sockets"])
    )

    print(genus_summary)


if __name__ == "__main__":
    main()

