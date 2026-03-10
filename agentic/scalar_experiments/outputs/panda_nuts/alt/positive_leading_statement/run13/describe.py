import json
import pandas as pd


df = pd.read_csv("panda_nuts.csv")
df["efficiency"] = df["nuts_opened"] / df["seconds"]

summary = {
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_median": float(df["efficiency"].median()),
    "efficiency_std": float(df["efficiency"].std()),
    "mean_by_sex": df.groupby("sex")["efficiency"].mean().to_dict(),
    "mean_by_help": df.groupby("help")["efficiency"].mean().to_dict(),
}

with open("describe_results.json", "w") as f:
    json.dump(summary, f, indent=2)
