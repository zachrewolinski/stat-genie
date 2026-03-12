import pandas as pd

# Load
path = "soccer.csv"
df = pd.read_csv(path)

# skin tone
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_tone=skin)
sub = df.dropna(subset=["skin_tone"]).copy()

# Aggregate to player
player_agg = (
    sub.groupby("playerShort", as_index=False)
    .agg(skin_tone=("skin_tone", "mean"), games=("games", "sum"), redCards=("redCards", "sum"))
)

# Round to 2 decimals to see categories
player_agg["skin_round"] = player_agg["skin_tone"].round(2)
print(player_agg["skin_round"].value_counts().sort_index())

# Use median split
median = player_agg["skin_tone"].median()
print("median", median)

# quartiles
print(player_agg["skin_tone"].quantile([0.25,0.5,0.75]))
