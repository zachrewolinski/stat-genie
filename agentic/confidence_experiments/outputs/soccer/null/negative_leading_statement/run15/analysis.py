import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Skin tone: average of the two raters (normalized 0-1 scale).
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Basic filters
base = df[(df["games"] > 0) & df["skin_tone"].notna()].copy()

# Ensure counts are integers
base["redCards"] = pd.to_numeric(base["redCards"], errors="coerce")
base["games"] = pd.to_numeric(base["games"], errors="coerce")
base = base.dropna(subset=["redCards", "games", "skin_tone"])

# Poisson regression on dyads with offset for exposure (games)
poisson_model = smf.glm(
    "redCards ~ skin_tone",
    data=base,
    family=sm.families.Poisson(),
    offset=np.log(base["games"]),
).fit(cov_type="cluster", cov_kwds={"groups": base["playerShort"]})

# Adjusted model with basic covariates (position, league, height, weight, age)
# Parse birthday -> age in 2013 (season year end)
# Handle missing values by dropping rows in adjusted model
birthday = pd.to_datetime(base["birthday"], format="%d.%m.%Y", errors="coerce")
base["age_2013"] = 2013 - birthday.dt.year

adj_cols = [
    "redCards",
    "games",
    "skin_tone",
    "position",
    "leagueCountry",
    "height",
    "weight",
    "age_2013",
    "playerShort",
]
adj = base[adj_cols].dropna()

adj_model = smf.glm(
    "redCards ~ skin_tone + C(position) + C(leagueCountry) + height + weight + age_2013",
    data=adj,
    family=sm.families.Poisson(),
    offset=np.log(adj["games"]),
).fit(cov_type="cluster", cov_kwds={"groups": adj["playerShort"]})

# Player-level aggregation for a simple rate comparison
player = base.groupby("playerShort").agg(
    skin_tone=("skin_tone", "mean"),
    redCards=("redCards", "sum"),
    games=("games", "sum"),
)
player["rate"] = player["redCards"] / player["games"]

# Define light vs dark using extremes to sharpen contrast
light = player[player["skin_tone"] <= 0.25].copy()
dark = player[player["skin_tone"] >= 0.75].copy()

# Poisson regression at player level with binary dark indicator
player_bin = player[(player["skin_tone"] <= 0.25) | (player["skin_tone"] >= 0.75)].copy()
player_bin["dark"] = (player_bin["skin_tone"] >= 0.75).astype(int)

player_model = smf.glm(
    "redCards ~ dark",
    data=player_bin,
    family=sm.families.Poisson(),
    offset=np.log(player_bin["games"]),
).fit(cov_type="HC3")

results = {
    "n_dyads": int(len(base)),
    "n_players": int(player.shape[0]),
    "poisson_coef": float(poisson_model.params["skin_tone"]),
    "poisson_p": float(poisson_model.pvalues["skin_tone"]),
    "poisson_rr": float(np.exp(poisson_model.params["skin_tone"])),
    "poisson_ci": [float(np.exp(ci)) for ci in poisson_model.conf_int().loc["skin_tone"].tolist()],
    "adj_coef": float(adj_model.params["skin_tone"]),
    "adj_p": float(adj_model.pvalues["skin_tone"]),
    "adj_rr": float(np.exp(adj_model.params["skin_tone"])),
    "adj_ci": [float(np.exp(ci)) for ci in adj_model.conf_int().loc["skin_tone"].tolist()],
    "player_light_n": int(light.shape[0]),
    "player_dark_n": int(dark.shape[0]),
    "player_light_rate": float(light["rate"].mean()) if light.shape[0] > 0 else None,
    "player_dark_rate": float(dark["rate"].mean()) if dark.shape[0] > 0 else None,
    "player_model_coef": float(player_model.params["dark"]),
    "player_model_p": float(player_model.pvalues["dark"]),
    "player_model_rr": float(np.exp(player_model.params["dark"])),
    "player_model_ci": [float(np.exp(ci)) for ci in player_model.conf_int().loc["dark"].tolist()],
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
