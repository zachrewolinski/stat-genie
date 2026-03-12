import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute average skin tone per row; ratings in [0,1] with 5 discrete steps
skin = df[["rater1", "rater2"]].mean(axis=1)

df = df.assign(skin=skin)

# Aggregate to player level to avoid player-referee dyad duplication
player_agg = (
    df.dropna(subset=["skin"])  # keep players with skin ratings
      .groupby("playerShort", as_index=False)
      .agg(
          skin=("skin", "first"),
          games=("games", "sum"),
          redCards=("redCards", "sum"),
      )
)

# Guard against zero games
player_agg = player_agg[player_agg["games"] > 0].copy()

# Define dark/light groups based on 5-point scale normalized to 0..1
# 0.0, 0.25, 0.5, 0.75, 1.0; use <=0.25 as light, >=0.75 as dark
player_agg["skin_group"] = pd.cut(
    player_agg["skin"],
    bins=[-np.inf, 0.25, 0.75, np.inf],
    labels=["light", "medium", "dark"],
)

light = player_agg[player_agg["skin_group"] == "light"]
dark = player_agg[player_agg["skin_group"] == "dark"]

# Rate per 100 games
light_rate = light["redCards"].sum() / light["games"].sum() * 100 if len(light) else np.nan
dark_rate = dark["redCards"].sum() / dark["games"].sum() * 100 if len(dark) else np.nan

# Poisson regression with exposure (games) and continuous skin tone
# y ~ skin + offset(log(games))
X = sm.add_constant(player_agg["skin"].astype(float))
y = player_agg["redCards"].astype(float)
offset = np.log(player_agg["games"].astype(float))

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type="HC3")

beta_skin = poisson_res.params["skin"]
se_skin = poisson_res.bse["skin"]
p_skin = poisson_res.pvalues["skin"]
rr_skin = float(np.exp(beta_skin))

# Compare dark vs light via rate ratio (Poisson test approximation)
# Compute log rate ratio and SE
# If either group missing, skip
rate_ratio = None
rate_ratio_p = None
if len(light) and len(dark):
    # Poisson counts with exposure; log rate ratio = log(dark_rate/light_rate)
    # Approx SE = sqrt(1/red_dark + 1/red_light)
    red_dark = dark["redCards"].sum()
    red_light = light["redCards"].sum()
    if red_dark > 0 and red_light > 0:
        rate_ratio = (red_dark / dark["games"].sum()) / (red_light / light["games"].sum())
        se_log = np.sqrt(1.0 / red_dark + 1.0 / red_light)
        z = np.log(rate_ratio) / se_log
        rate_ratio_p = 2 * (1 - norm.cdf(abs(z)))

summary = {
    "n_players": int(player_agg.shape[0]),
    "n_light": int(light.shape[0]),
    "n_dark": int(dark.shape[0]),
    "light_rate_per_100": float(light_rate) if not np.isnan(light_rate) else None,
    "dark_rate_per_100": float(dark_rate) if not np.isnan(dark_rate) else None,
    "poisson_beta_skin": float(beta_skin),
    "poisson_se_skin": float(se_skin),
    "poisson_p_skin": float(p_skin),
    "poisson_rr_skin": rr_skin,
    "rate_ratio_dark_vs_light": float(rate_ratio) if rate_ratio is not None else None,
    "rate_ratio_p": float(rate_ratio_p) if rate_ratio_p is not None else None,
}

print(json.dumps(summary, indent=2))
