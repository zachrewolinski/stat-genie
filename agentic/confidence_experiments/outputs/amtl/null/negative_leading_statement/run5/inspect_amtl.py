from pathlib import Path

import pandas as pd


def main():
    df_raw = pd.read_csv(Path("amtl.csv"))
    print("Raw shape:", df_raw.shape)
    print("Columns:", df_raw.columns.tolist())

    bad_ratio = df_raw[df_raw["num_amtl"] > df_raw["sockets"]]
    print("Rows where num_amtl > sockets:", bad_ratio.shape[0])

    zero_sockets = df_raw[df_raw["sockets"] <= 0]
    print("Rows with sockets <= 0:", zero_sockets.shape[0])

    missing_any = df_raw[
        df_raw[["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus", "specimen"]].isna().any(
            axis=1
        )
    ]
    print("Rows with missing in key fields:", missing_any.shape[0])

    print("Raw num_amtl summary:")
    print(df_raw["num_amtl"].describe())
    print("Raw sockets summary:")
    print(df_raw["sockets"].describe())

    # Apply the same cleaning steps as in analyze_amtl.py
    invalid_ratio_mask = df_raw["num_amtl"] > df_raw["sockets"]
    df = df_raw.loc[~invalid_ratio_mask].copy()
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "tooth_class",
            "genus",
            "specimen",
        ]
    )
    df = df[df["sockets"] > 0]
    df["is_human"] = df["genus"].astype(str).str.startswith("Homo").astype(int)

    print("Cleaned shape:", df.shape)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    print("Proportion summary (num_amtl / sockets):")
    print(df["prop_amtl"].describe())
    print("Min proportion:", df["prop_amtl"].min(), "Max proportion:", df["prop_amtl"].max())



if __name__ == "__main__":
    main()
