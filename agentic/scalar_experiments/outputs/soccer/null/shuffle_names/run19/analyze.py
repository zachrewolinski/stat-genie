import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Column mapping per info.json descriptions
# Skin tone ratings: rater1 and nExp (both normalized 0-1)
skin_cols = ["rater1", "nExp"]
# Red cards count: column described as 'Number of red cards player received from referee.'
red_col = "yellowCards"
# Games exposure: column described as 'Number of games in the player-referee dyad.'
games_col = "redCards"
# Player identifier
player_id_col = "photoID"

# Basic sanity checks
missing_cols = [c for c in skin_cols + [red_col, games_col, player_id_col] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# Coerce numeric
for c in skin_cols + [red_col, games_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute mean skin tone per row
skin = df[skin_cols].mean(axis=1, skipna=True)
df["skin_mean"] = skin

# Drop rows with missing core values
core = df[["skin_mean", red_col, games_col, player_id_col]].copy()
core = core.dropna(subset=["skin_mean", red_col, games_col, player_id_col])
core = core[core[games_col] > 0]

# Aggregate to player level to reduce dyad dependence
player = (
    core.groupby(player_id_col)
    .agg(
        total_red_cards=(red_col, "sum"),
        total_games=(games_col, "sum"),
        skin_mean=("skin_mean", "mean"),
        dyads=(games_col, "count"),
    )
    .reset_index()
)
player = player[player["total_games"] > 0]

player["red_rate"] = player["total_red_cards"] / player["total_games"]

# Define dark vs light based on midpoint 0.5 (normalized 0-1)
player["dark"] = player["skin_mean"] > 0.5

# Group comparison
summary = player.groupby("dark").agg(
    players=(player_id_col, "count"),
    mean_skin=("skin_mean", "mean"),
    mean_red_rate=("red_rate", "mean"),
    mean_red_cards=("total_red_cards", "mean"),
    mean_games=("total_games", "mean"),
)

# Poisson regression at player level with offset(log games)
X = sm.add_constant(player["skin_mean"])
model_poisson = sm.GLM(
    player["total_red_cards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(player["total_games"]),
)
res_poisson = model_poisson.fit(cov_type="HC3")

# Negative binomial as robustness
model_nb = sm.GLM(
    player["total_red_cards"],
    X,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(player["total_games"]),
)
res_nb = model_nb.fit(cov_type="HC3")

# Dyad-level Poisson with offset
X_d = sm.add_constant(core["skin_mean"])
model_poisson_d = sm.GLM(
    core[red_col],
    X_d,
    family=sm.families.Poisson(),
    offset=np.log(core[games_col]),
)
res_poisson_d = model_poisson_d.fit(cov_type="HC3")

# Extract key stats

def coef_info(res):
    coef = res.params["skin_mean"]
    se = res.bse["skin_mean"]
    p = res.pvalues["skin_mean"]
    irr = np.exp(coef)
    return coef, se, p, irr

coef_p, se_p, p_p, irr_p = coef_info(res_poisson)
coef_nb, se_nb, p_nb, irr_nb = coef_info(res_nb)
coef_pd, se_pd, p_pd, irr_pd = coef_info(res_poisson_d)

# Simple correlation
corr = player["skin_mean"].corr(player["red_rate"], method="pearson")

# Print results
print("Player-level summary (dark vs light):")
print(summary)
print()
print("Player-level Poisson (offset log games) skin_mean:")
print({"coef": coef_p, "se": se_p, "p": p_p, "irr": irr_p})
print()
print("Player-level NegBin (offset log games) skin_mean:")
print({"coef": coef_nb, "se": se_nb, "p": p_nb, "irr": irr_nb})
print()
print("Dyad-level Poisson (offset log games) skin_mean:")
print({"coef": coef_pd, "se": se_pd, "p": p_pd, "irr": irr_pd})
print()
print("Player-level corr(skin_mean, red_rate):", corr)

# Save key results to CSV for later if needed
result = {
    "players": int(player.shape[0]),
    "dyads": int(core.shape[0]),
    "corr": float(corr),
    "poisson_player_coef": float(coef_p),
    "poisson_player_p": float(p_p),
    "poisson_player_irr": float(irr_p),
    "nb_player_coef": float(coef_nb),
    "nb_player_p": float(p_nb),
    "nb_player_irr": float(irr_nb),
    "poisson_dyad_coef": float(coef_pd),
    "poisson_dyad_p": float(p_pd),
    "poisson_dyad_irr": float(irr_pd),
}

pd.DataFrame([result]).to_csv("analysis_results.csv", index=False)
