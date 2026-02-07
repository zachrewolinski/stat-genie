import pandas as pd

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

for c in ["missing_teeth", "observable_sockets", "age", "sex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

mask_obs = df["observable_sockets"] > 0
mask_miss = df["missing_teeth"] >= 0
mask_leq = df["missing_teeth"] <= df["observable_sockets"]

print("observable<=0", (~mask_obs).sum())
print("missing<0", (~mask_miss).sum())
print("missing>observable", (~mask_leq).sum())

print(df.loc[~mask_leq, ["missing_teeth","observable_sockets"]].head())
