import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone: average of rater1 and rater2
# Some rows may have missing; use mean across available raters
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_tone=skin)

# Filter to rows with skin tone and valid games/redCards
# games might be zero? per metadata min 1; still filter >0
sub = df.dropna(subset=["skin_tone", "games", "redCards"])
sub = sub[sub["games"] > 0].copy()

# Aggregate to player level to avoid repeated measures and to align with player-level skin tone
player_agg = (
    sub.groupby(["playerShort"], as_index=False)
    .agg(
        skin_tone=("skin_tone", "mean"),
        games=("games", "sum"),
        redCards=("redCards", "sum"),
    )
)

# Also compute light/dark groups based on skin tone median or thresholds
# Here use conventional split: light <= 0.5, dark > 0.5 (since 5-point normalized)
player_agg["dark"] = (player_agg["skin_tone"] > 0.5).astype(int)

# Descriptive stats
n_players = len(player_agg)

# Rates per game
player_agg["red_rate"] = player_agg["redCards"] / player_agg["games"]

rate_light = player_agg.loc[player_agg.dark == 0, "red_rate"].mean()
rate_dark = player_agg.loc[player_agg.dark == 1, "red_rate"].mean()

# Poisson regression with offset for exposure
# Outcome: redCards (count); predictor: skin_tone (continuous)
# Use robust (HC1) standard errors
X = sm.add_constant(player_agg[["skin_tone"]])
model = sm.GLM(
    player_agg["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(player_agg["games"]),
)
res = model.fit(cov_type="HC1")

# Also fit model with binary dark indicator
X2 = sm.add_constant(player_agg[["dark"]])
model2 = sm.GLM(
    player_agg["redCards"],
    X2,
    family=sm.families.Poisson(),
    offset=np.log(player_agg["games"]),
)
res2 = model2.fit(cov_type="HC1")

# Collect key results
results = {
    "n_players": n_players,
    "rate_light": rate_light,
    "rate_dark": rate_dark,
    "coef_skin": res.params["skin_tone"],
    "p_skin": res.pvalues["skin_tone"],
    "coef_dark": res2.params["dark"],
    "p_dark": res2.pvalues["dark"],
}

# Save results for inspection
pd.Series(results).to_csv("analysis_results.csv")

print(results)
