import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load
path = "soccer.csv"
df = pd.read_csv(path)

df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
sub = df.dropna(subset=["skin_tone", "games", "redCards"])
sub = sub[sub["games"] > 0].copy()

# Aggregate per player
player_agg = (
    sub.groupby(["playerShort"], as_index=False)
    .agg(
        skin_tone=("skin_tone", "mean"),
        games=("games", "sum"),
        redCards=("redCards", "sum"),
    )
)
player_agg["dark"] = (player_agg["skin_tone"] > 0.5).astype(int)

# totals
summary = player_agg.groupby("dark").agg(
    players=("playerShort", "count"),
    games=("games", "sum"),
    redCards=("redCards", "sum"),
)
summary["rate"] = summary["redCards"] / summary["games"]
print(summary)

# Check proportion of players with >=1 red card in each group
summary2 = player_agg.groupby("dark").agg(
    players=("playerShort", "count"),
    any_red=("redCards", lambda x: (x>0).mean()),
    mean_red=("redCards", "mean"),
)
print(summary2)

# Fit Poisson models with robust SE
X = sm.add_constant(player_agg[["skin_tone"]])
model = sm.GLM(
    player_agg["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(player_agg["games"]),
)
res = model.fit(cov_type="HC1")
print(res.summary())

X2 = sm.add_constant(player_agg[["dark"]])
model2 = sm.GLM(
    player_agg["redCards"],
    X2,
    family=sm.families.Poisson(),
    offset=np.log(player_agg["games"]),
)
res2 = model2.fit(cov_type="HC1")
print(res2.summary())
