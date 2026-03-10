import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Compute skin tone per row (mean of rater1 and rater2)
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin tone and games
usable = df[df["skin_tone"].notna() & df["games"].notna()].copy()

# Aggregate to player level
agg = (
    usable.groupby("playerShort")
    .agg(
        skin_tone=("skin_tone", "mean"),
        redCards=("redCards", "sum"),
        games=("games", "sum"),
    )
    .reset_index()
)
agg = agg[agg["games"] > 0].copy()

# Distribution of skin tone values (rounded for readability)
skin_counts = agg["skin_tone"].round(3).value_counts().sort_index()

# Player-level Poisson regression with exposure (games) and robust SE
X_player = sm.add_constant(agg["skin_tone"])
y_player = agg["redCards"]
model_player = sm.GLM(
    y_player,
    X_player,
    family=sm.families.Poisson(),
    offset=np.log(agg["games"]),
)
res_player = model_player.fit(cov_type="HC1")

beta_player = res_player.params["skin_tone"]
p_player = res_player.pvalues["skin_tone"]
ci_player = res_player.conf_int().loc["skin_tone"].tolist()
irr_player = float(np.exp(beta_player))
irr_ci_player = [float(np.exp(ci_player[0])), float(np.exp(ci_player[1]))]

player_dispersion = float(res_player.pearson_chi2 / res_player.df_resid) if res_player.df_resid > 0 else float("nan")

# Dyad-level Poisson regression with player-clustered SE
X_dyad = sm.add_constant(usable["skin_tone"])
y_dyad = usable["redCards"]
model_dyad = sm.GLM(
    y_dyad,
    X_dyad,
    family=sm.families.Poisson(),
    offset=np.log(usable["games"]),
)
res_dyad = model_dyad.fit(cov_type="cluster", cov_kwds={"groups": usable["playerShort"]})

beta_dyad = res_dyad.params["skin_tone"]
p_dyad = res_dyad.pvalues["skin_tone"]
ci_dyad = res_dyad.conf_int().loc["skin_tone"].tolist()
irr_dyad = float(np.exp(beta_dyad))
irr_ci_dyad = [float(np.exp(ci_dyad[0])), float(np.exp(ci_dyad[1]))]

# Group comparisons with alternative thresholds
thresholds = {
    "light_le_0.25_dark_ge_0.75": (0.25, 0.75),
    "light_le_0.25_dark_ge_0.5": (0.25, 0.5),
    "light_le_0.5_dark_gt_0.5": (0.5, 0.5),
}

group_stats = {}
for name, (light_thr, dark_thr) in thresholds.items():
    if name == "light_le_0.5_dark_gt_0.5":
        light = agg[agg["skin_tone"] <= light_thr]
        dark = agg[agg["skin_tone"] > dark_thr]
    else:
        light = agg[agg["skin_tone"] <= light_thr]
        dark = agg[agg["skin_tone"] >= dark_thr]
    light_red = light["redCards"].sum()
    light_games = light["games"].sum()
    dark_red = dark["redCards"].sum()
    dark_games = dark["games"].sum()
    light_rate = float(light_red / light_games) if light_games > 0 else float("nan")
    dark_rate = float(dark_red / dark_games) if dark_games > 0 else float("nan")
    rate_ratio = float(dark_rate / light_rate) if light_rate > 0 else float("inf")
    group_stats[name] = {
        "light_rate_per_game": light_rate,
        "dark_rate_per_game": dark_rate,
        "rate_ratio": rate_ratio,
        "light_n_players": int(light.shape[0]),
        "dark_n_players": int(dark.shape[0]),
    }

# Skin-tone quintiles and rate per game
agg["skin_quintile"] = pd.qcut(agg["skin_tone"], 5, labels=False, duplicates="drop")
quintile_rates = (
    agg.groupby("skin_quintile")
    .apply(lambda g: g["redCards"].sum() / g["games"].sum())
    .astype(float)
    .to_dict()
)

n_players = int(agg.shape[0])

explanation = {
    "n_players_with_skin_tone": n_players,
    "skin_tone_value_counts": {str(k): int(v) for k, v in skin_counts.items()},
    "player_level_poisson": {
        "irr_per_1_unit_skin_tone": irr_player,
        "irr_95ci": irr_ci_player,
        "p_value": float(p_player),
        "dispersion": player_dispersion,
    },
    "dyad_level_poisson_clustered": {
        "irr_per_1_unit_skin_tone": irr_dyad,
        "irr_95ci": irr_ci_dyad,
        "p_value": float(p_dyad),
    },
    "group_comparisons": group_stats,
    "skin_quintile_redcard_rate_per_game": {str(k): float(v) for k, v in quintile_rates.items()},
}

with open("analysis_results.json", "w") as f:
    json.dump(explanation, f, indent=2)

print(json.dumps(explanation, indent=2))
