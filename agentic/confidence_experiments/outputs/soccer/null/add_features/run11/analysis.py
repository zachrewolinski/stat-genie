import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Create skin tone average (normalized 0-1)
# Use mean of rater1 and rater2 when both present; if one missing, use the other.
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Add skin tone to df
_df = df.copy()
_df["skin"] = skin

# Keep rows with skin tone and games > 0
_df = _df[_df["skin"].notna()]
_df = _df[_df["games"] > 0]

# Aggregate per player
player_cols = ["playerShort", "skin"]
player_agg = (
    _df.groupby("playerShort", as_index=False)
    .agg(
        skin=("skin", "mean"),
        redCards=("redCards", "sum"),
        games=("games", "sum"),
    )
)
player_agg = player_agg[player_agg["games"] > 0]

# Rate per 100 games
player_agg["red_rate_per_100"] = player_agg["redCards"] / player_agg["games"] * 100

# Define light/dark groups using thirds
light_threshold = player_agg["skin"].quantile(1/3)
dark_threshold = player_agg["skin"].quantile(2/3)
player_agg["skin_group"] = np.where(
    player_agg["skin"] <= light_threshold,
    "light",
    np.where(player_agg["skin"] >= dark_threshold, "dark", "medium"),
)

# Summaries for light vs dark
summary_groups = (
    player_agg[player_agg["skin_group"].isin(["light", "dark"])]
    .groupby("skin_group")
    .agg(
        players=("playerShort", "count"),
        total_red=("redCards", "sum"),
        total_games=("games", "sum"),
    )
)
summary_groups["rate_per_100"] = summary_groups["total_red"] / summary_groups["total_games"] * 100

# Poisson regression on player-aggregated data: redCards ~ skin (continuous)
X = sm.add_constant(player_agg["skin"])
model = sm.GLM(
    player_agg["redCards"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(player_agg["games"]),
)
result = model.fit(cov_type="HC3")

# Poisson regression for light vs dark only, indicator for dark
ld = player_agg[player_agg["skin_group"].isin(["light", "dark"])].copy()
ld["dark"] = (ld["skin_group"] == "dark").astype(int)
X_ld = sm.add_constant(ld["dark"])
model_ld = sm.GLM(
    ld["redCards"],
    X_ld,
    family=sm.families.Poisson(),
    offset=np.log(ld["games"]),
)
result_ld = model_ld.fit(cov_type="HC3")

# Extract stats
coef_skin = result.params["skin"]
se_skin = result.bse["skin"]
p_skin = result.pvalues["skin"]
rate_ratio_skin = float(np.exp(coef_skin))

coef_dark = result_ld.params["dark"]
se_dark = result_ld.bse["dark"]
p_dark = result_ld.pvalues["dark"]
rate_ratio_dark = float(np.exp(coef_dark))

results = {
    "n_rows": int(len(df)),
    "n_rows_with_skin": int(len(_df)),
    "n_players_with_skin": int(len(player_agg)),
    "light_threshold": float(light_threshold),
    "dark_threshold": float(dark_threshold),
    "group_summary": summary_groups.reset_index().to_dict(orient="records"),
    "poisson_continuous": {
        "coef_skin": float(coef_skin),
        "se_skin": float(se_skin),
        "p_value": float(p_skin),
        "rate_ratio_per_1unit": rate_ratio_skin,
    },
    "poisson_dark_vs_light": {
        "coef_dark": float(coef_dark),
        "se_dark": float(se_dark),
        "p_value": float(p_dark),
        "rate_ratio_dark_vs_light": rate_ratio_dark,
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
