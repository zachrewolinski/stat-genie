import pandas as pd

COL_SKIN1 = "feature18"
COL_SKIN2 = "feature19"

df = pd.read_csv("soccer.csv")
df["skin_avg"] = df[[COL_SKIN1, COL_SKIN2]].mean(axis=1)
vc = df["skin_avg"].value_counts(dropna=False).sort_index()
print(vc)
