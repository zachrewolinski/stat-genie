import pandas as pd
import numpy as np

col_to_name = {
    "feature1": "tooth_class",
    "feature2": "specimen_id",
    "feature3": "missing_teeth",
    "feature4": "observable_sockets",
    "feature5": "age",
    "feature6": "age_uncertainty",
    "feature7": "sex",
    "feature8": "genus",
    "feature9": "region",
}

df = pd.read_csv("amtl.csv").rename(columns=col_to_name)

needed = ["tooth_class", "missing_teeth", "observable_sockets", "age", "sex", "genus"]
print("rows", len(df))

for c in ["missing_teeth", "observable_sockets", "age", "sex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("na counts", df[needed].isna().sum())

mask_valid = (df["observable_sockets"] > 0) & (df["missing_teeth"] >= 0) & (df["missing_teeth"] <= df["observable_sockets"])
print("valid rows", mask_valid.sum())

print("observable min", df["observable_sockets"].min(), "max", df["observable_sockets"].max())
print("missing min", df["missing_teeth"].min(), "max", df["missing_teeth"].max())

invalid = df[~mask_valid]
print("invalid head")
print(invalid.head())

# check non-integer observable
print("non-integer observable", (df["observable_sockets"] % 1 != 0).sum())
print("non-integer missing", (df["missing_teeth"] % 1 != 0).sum())
