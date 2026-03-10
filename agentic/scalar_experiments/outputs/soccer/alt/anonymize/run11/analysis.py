import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Map columns
player_col = "feature1"
red_col = "feature16"
games_col = "feature9"
skin1 = "feature18"
skin2 = "feature19"

# Compute mean skin tone and filter
skin_mean = df[[skin1, skin2]].mean(axis=1, skipna=True)

df = df.assign(skin_mean=skin_mean)

# Keep valid rows
valid = df[skin_mean.notna() & df[games_col].notna() & df[red_col].notna()]
valid = valid[valid[games_col] > 0]

# Aggregate to player level
player = (
    valid.groupby(player_col)
    .agg(
        skin_mean=("skin_mean", "mean"),
        games=(games_col, "sum"),
        red_cards=(red_col, "sum"),
    )
    .reset_index()
)

# Drop any players with zero games after aggregation (shouldn't happen)
player = player[player["games"] > 0]

# Rates by group (light vs dark)
light_mask = player["skin_mean"] <= 0.25
dark_mask = player["skin_mean"] >= 0.75

light = player[light_mask].copy()
dark = player[dark_mask].copy()

def rate_summary(df_in: pd.DataFrame) -> dict:
    games = df_in["games"].sum()
    reds = df_in["red_cards"].sum()
    rate = reds / games if games > 0 else np.nan
    return {"games": int(games), "red_cards": int(reds), "rate": float(rate)}

summary = {
    "n_players_total": int(player.shape[0]),
    "n_players_light": int(light.shape[0]),
    "n_players_dark": int(dark.shape[0]),
    "light": rate_summary(light),
    "dark": rate_summary(dark),
}

# Poisson regression at player level with offset log(games)
player["log_games"] = np.log(player["games"])
X_cont = sm.add_constant(player["skin_mean"])
model_cont = sm.GLM(player["red_cards"], X_cont, family=sm.families.Poisson(), offset=player["log_games"])
res_cont = model_cont.fit()

# Binary comparison: dark vs light only
dl = player[light_mask | dark_mask].copy()
dl["dark"] = (dl["skin_mean"] >= 0.75).astype(int)
dl["log_games"] = np.log(dl["games"])
X_bin = sm.add_constant(dl["dark"])
model_bin = sm.GLM(dl["red_cards"], X_bin, family=sm.families.Poisson(), offset=dl["log_games"])
res_bin = model_bin.fit()

# Extract results
cont_coef = float(res_cont.params["skin_mean"])
cont_p = float(res_cont.pvalues["skin_mean"])
cont_irr = float(np.exp(cont_coef))

bin_coef = float(res_bin.params["dark"])
bin_p = float(res_bin.pvalues["dark"])
bin_irr = float(np.exp(bin_coef))

output = {
    "summary": summary,
    "cont_irr": cont_irr,
    "cont_p": cont_p,
    "bin_irr": bin_irr,
    "bin_p": bin_p,
}

print(json.dumps(output, indent=2))
