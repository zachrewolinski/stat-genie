import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Compute average skin tone (0=very light, 1=very dark)
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

# Drop rows without skin ratings or games
base = df.dropna(subset=["skin_tone", "games", "redCards"]).copy()
base = base[base["games"] > 0]

# Define light/dark groups using extreme bins to avoid ambiguous mid-tones
light = base[base["skin_tone"] <= 0.25]
dark = base[base["skin_tone"] >= 0.75]

# Compute simple rate comparisons (per 100 games)
def rate_per_100(g):
    return (g["redCards"].sum() / g["games"].sum()) * 100.0

light_rate = rate_per_100(light) if len(light) else np.nan
dark_rate = rate_per_100(dark) if len(dark) else np.nan

# Dyad-level Poisson regression with exposure (games)
# Using robust (HC1) SEs for mild overdispersion
base["log_games"] = np.log(base["games"])
model_dyad = smf.glm(
    formula="redCards ~ skin_tone",
    data=base,
    family=sm.families.Poisson(),
    offset=base["log_games"],
).fit(cov_type="HC1")

# Player-level aggregation to reduce repeated-measures concerns
player = base.groupby("playerShort", as_index=False).agg(
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    skin_tone=("skin_tone", "mean"),
)
player = player[player["total_games"] > 0]
player["log_games"] = np.log(player["total_games"])

model_player = smf.glm(
    formula="total_red ~ skin_tone",
    data=player,
    family=sm.families.Poisson(),
    offset=player["log_games"],
).fit(cov_type="HC1")

# Collect key outputs
summary = {
    "n_dyads": int(len(base)),
    "n_players": int(len(player)),
    "light_rate_per_100_games": float(light_rate),
    "dark_rate_per_100_games": float(dark_rate),
    "dyad_skin_coef": float(model_dyad.params["skin_tone"]),
    "dyad_skin_p": float(model_dyad.pvalues["skin_tone"]),
    "player_skin_coef": float(model_player.params["skin_tone"]),
    "player_skin_p": float(model_player.pvalues["skin_tone"]),
}

print(pd.Series(summary))
