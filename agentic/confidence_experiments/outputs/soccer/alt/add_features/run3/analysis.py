import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Create mean skin tone from rater1 and rater2
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_mean=skin)

# Keep rows with skin info and games > 0
base = df.dropna(subset=["skin_mean", "games", "redCards"]).copy()
base = base[base["games"] > 0]

# Binary dark/light categories
light_mask = base["skin_mean"] <= 0.25
mid_mask = (base["skin_mean"] > 0.25) & (base["skin_mean"] < 0.75)
dark_mask = base["skin_mean"] >= 0.75

# Summary counts
summary = {
    "n_total": len(base),
    "n_light": int(light_mask.sum()),
    "n_mid": int(mid_mask.sum()),
    "n_dark": int(dark_mask.sum()),
}

# Rates per game by group
rate_light = base.loc[light_mask, "redCards"].sum() / base.loc[light_mask, "games"].sum()
rate_dark = base.loc[dark_mask, "redCards"].sum() / base.loc[dark_mask, "games"].sum()

summary.update({
    "rate_light": rate_light,
    "rate_dark": rate_dark,
    "rate_ratio_dark_vs_light": rate_dark / rate_light if rate_light > 0 else np.nan,
})

# Poisson regression with offset, dark vs light only
poisson_df = base.loc[light_mask | dark_mask].copy()
poisson_df["dark"] = (poisson_df["skin_mean"] >= 0.75).astype(int)
poisson_df["log_games"] = np.log(poisson_df["games"])

# Unclustered Poisson
model = smf.glm(
    "redCards ~ dark",
    data=poisson_df,
    family=sm.families.Poisson(),
    offset=poisson_df["log_games"],
).fit()

# Cluster-robust by playerShort to account for repeated dyads
clustered = smf.glm(
    "redCards ~ dark",
    data=poisson_df,
    family=sm.families.Poisson(),
    offset=poisson_df["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": poisson_df["playerShort"]})

# Continuous skin_mean
cont_model = smf.glm(
    "redCards ~ skin_mean",
    data=base,
    family=sm.families.Poisson(),
    offset=np.log(base["games"]),
).fit(cov_type="cluster", cov_kwds={"groups": base["playerShort"]})

# Player-level aggregation to reduce dependence
player = base.groupby("playerShort").agg(
    redCards_sum=("redCards", "sum"),
    games_sum=("games", "sum"),
    skin_mean=("skin_mean", "mean"),
).reset_index()
player = player[player["games_sum"] > 0]

player["dark"] = (player["skin_mean"] >= 0.75).astype(int)
player["light"] = (player["skin_mean"] <= 0.25).astype(int)
player_dl = player[(player["dark"] == 1) | (player["light"] == 1)].copy()
player_dl["log_games"] = np.log(player_dl["games_sum"])

player_model = smf.glm(
    "redCards_sum ~ dark",
    data=player_dl,
    family=sm.families.Poisson(),
    offset=player_dl["log_games"],
).fit()

# Output results
print("SUMMARY", summary)
print("POISSON_DARK_COEF", model.params["dark"], "SE", model.bse["dark"], "P", model.pvalues["dark"])
print("POISSON_DARK_COEF_CLUSTER", clustered.params["dark"], "SE", clustered.bse["dark"], "P", clustered.pvalues["dark"])
print("CONT_SKIN_COEF_CLUSTER", cont_model.params["skin_mean"], "SE", cont_model.bse["skin_mean"], "P", cont_model.pvalues["skin_mean"])
print("PLAYER_LEVEL_DARK_COEF", player_model.params["dark"], "SE", player_model.bse["dark"], "P", player_model.pvalues["dark"])

# Also compute logistic any red card with offset not possible; use binomial by probability of red per game? We'll skip here
