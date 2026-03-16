import pandas as pd
import numpy as np

csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

skin_cols = ["rater1", "nExp"]
red_col = "yellowCards"
games_col = "redCards"
player_id_col = "photoID"

for c in skin_cols + [red_col, games_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["skin_mean"] = df[skin_cols].mean(axis=1, skipna=True)
core = df[["skin_mean", red_col, games_col, player_id_col]].dropna()
core = core[core[games_col] > 0]

player = (
    core.groupby(player_id_col)
    .agg(
        total_red_cards=(red_col, "sum"),
        total_games=(games_col, "sum"),
        skin_mean=("skin_mean", "mean"),
    )
    .reset_index()
)
player = player[player["total_games"] > 0]
player["red_rate"] = player["total_red_cards"] / player["total_games"]

# Skin tone distribution
print("Skin_mean summary:")
print(player["skin_mean"].describe())
print("Unique skin_mean values:", sorted(player["skin_mean"].unique())[:10], "... total", player["skin_mean"].nunique())

# Quintiles for dark/light comparison
player["quintile"] = pd.qcut(player["skin_mean"], 5, labels=False, duplicates="drop")
q_summary = player.groupby("quintile").agg(
    players=(player_id_col, "count"),
    mean_skin=("skin_mean", "mean"),
    mean_red_rate=("red_rate", "mean"),
    mean_red_cards=("total_red_cards", "mean"),
    mean_games=("total_games", "mean"),
)
print("\nQuintile summary (0=lightest,4=darkest):")
print(q_summary)

# Top vs bottom quintile difference
if player["quintile"].notna().any():
    light = player[player["quintile"] == player["quintile"].min()]
    dark = player[player["quintile"] == player["quintile"].max()]
    diff_rate = dark["red_rate"].mean() - light["red_rate"].mean()
    print("\nTop vs bottom quintile red_rate diff:", diff_rate)
