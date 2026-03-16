import pandas as pd

COL_PLAYER = "feature1"
COL_SKIN1 = "feature18"
COL_SKIN2 = "feature19"
COL_GAMES = "feature9"
COL_RED = "feature16"

df = pd.read_csv("soccer.csv")
df["skin_avg"] = df[[COL_SKIN1, COL_SKIN2]].mean(axis=1)
df = df[[COL_PLAYER, "skin_avg", COL_GAMES, COL_RED]].dropna()
player = (
    df.groupby(COL_PLAYER, as_index=False)
      .agg(skin_avg=("skin_avg", "mean"), games=(COL_GAMES, "sum"), red=(COL_RED, "sum"))
)
print(player["skin_avg"].value_counts().sort_index())
print("n_players", len(player))
