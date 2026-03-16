import pandas as pd
import numpy as np

_df = pd.read_csv("panda_nuts.csv")
_df = _df.copy()
_df["seconds"] = pd.to_numeric(_df["seconds"], errors="coerce")
_df["nuts_opened"] = pd.to_numeric(_df["nuts_opened"], errors="coerce")
_df = _df.dropna(subset=["seconds", "nuts_opened", "age", "sex", "help"])
_df["rate"] = _df["nuts_opened"] / _df["seconds"]

summary = {
    "overall_rate_mean": float(_df["rate"].mean()),
    "overall_rate_median": float(_df["rate"].median()),
    "rate_by_sex": _df.groupby("sex")["rate"].agg(["mean", "median", "count"]).to_dict(),
    "rate_by_help": _df.groupby("help")["rate"].agg(["mean", "median", "count"]).to_dict(),
}

print(summary)
