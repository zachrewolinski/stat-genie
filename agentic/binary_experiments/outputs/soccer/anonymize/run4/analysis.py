import pandas as pd
import numpy as np

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute mean skin tone rating
skin = df[["feature18", "feature19"]].mean(axis=1)

# Assign skin category: light (<=0.25), dark (>=0.75); exclude middle
conditions = [skin <= 0.25, skin >= 0.75]
choices = ["light", "dark"]
cat = np.select(conditions, choices, default="mid")

df = df.assign(skin_mean=skin, skin_cat=cat)

# Use player short name as identifier
player_id = "feature1"

# Aggregate to player level
agg = (
    df.groupby(player_id)
    .agg(
        skin_cat=("skin_cat", "first"),
        skin_mean=("skin_mean", "first"),
        total_red=("feature16", "sum"),
        total_games=("feature9", "sum"),
    )
    .reset_index()
)

# Filter to light/dark and players with games
agg = agg[(agg["skin_cat"].isin(["light", "dark"])) & (agg["total_games"] > 0)]

# Overall group rates (weighted by games)
rates = (
    agg.groupby("skin_cat")[["total_red", "total_games"]]
    .sum()
    .assign(rate=lambda x: x["total_red"] / x["total_games"])
)

light_rate = rates.loc["light", "rate"] if "light" in rates.index else np.nan
dark_rate = rates.loc["dark", "rate"] if "dark" in rates.index else np.nan
rate_diff = dark_rate - light_rate

# Bootstrap CI over players within each group
rng = np.random.default_rng(42)
B = 2000

def bootstrap_rate(players_df):
    n = len(players_df)
    idx = rng.integers(0, n, size=n)
    sample = players_df.iloc[idx]
    total_red = sample["total_red"].sum()
    total_games = sample["total_games"].sum()
    return total_red / total_games if total_games > 0 else np.nan

light_players = agg[agg["skin_cat"] == "light"].reset_index(drop=True)
dark_players = agg[agg["skin_cat"] == "dark"].reset_index(drop=True)

boot_diffs = []
for _ in range(B):
    lr = bootstrap_rate(light_players)
    dr = bootstrap_rate(dark_players)
    boot_diffs.append(dr - lr)

boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.nanpercentile(boot_diffs, [2.5, 97.5])

# Print summary
print("Players per group:")
print(agg["skin_cat"].value_counts())
print("\nGroup rates (red cards per game):")
print(rates)
print(f"\nRate difference (dark - light): {rate_diff:.6f}")
print(f"95% bootstrap CI for difference: [{ci_low:.6f}, {ci_high:.6f}]")
