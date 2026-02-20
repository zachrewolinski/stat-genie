import pandas as pd


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    df["missing_rate"] = df["missing"] / df["sockets"]
    invalid = df["missing"] > df["sockets"]

    print("Rows:", len(df))
    print("Genera counts:")
    print(df["genus"].value_counts())
    print("\nMissing summary by genus (mean, std):")
    print(
        df.groupby("genus")
        .agg(
            mean_missing=("missing", "mean"),
            mean_sockets=("sockets", "mean"),
            mean_rate=("missing_rate", "mean"),
            std_rate=("missing_rate", "std"),
        )
        .reset_index()
    )

    print("\nInvalid rows where missing > sockets:", invalid.sum())
    if invalid.any():
        print(df.loc[invalid, ["genus", "tooth_class", "missing", "sockets"]].head(20))


if __name__ == "__main__":
    main()

