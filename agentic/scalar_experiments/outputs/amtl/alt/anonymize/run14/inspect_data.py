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
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    df = df[df["sockets"] > 0].copy()
    df["prop_missing"] = df["missing"] / df["sockets"]

    print(df[["missing", "sockets", "prop_missing"]].describe())
    print("\nAny missing > sockets:", bool((df["missing"] > df["sockets"]).any()))
    print("Num rows with prop 0:", int((df["prop_missing"] == 0).sum()))
    print("Num rows with prop 1:", int((df["prop_missing"] == 1).sum()))


if __name__ == "__main__":
    main()

