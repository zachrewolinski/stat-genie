import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Compute skin tone average
skin_cols = ["rater1", "rater2"]
df["skin"] = df[skin_cols].mean(axis=1, skipna=True)

# Aggregate to player level
player = (
    df.groupby("playerShort", as_index=False)
      .agg(
          skin=("skin", "mean"),
          games=("games", "sum"),
          redCards=("redCards", "sum"),
          yellowReds=("yellowReds", "sum"),
      )
)

# Remove players without skin ratings or zero games
player = player.dropna(subset=["skin"]).copy()
player = player[player["games"] > 0]

player["red_rate"] = player["redCards"] / player["games"]

# Define light/dark groups (exclude middle)
light = player[player["skin"] <= 0.4].copy()
dark = player[player["skin"] >= 0.6].copy()

# Summary stats
summary = {
    "n_players": len(player),
    "n_light": len(light),
    "n_dark": len(dark),
    "red_rate_light": light["red_rate"].mean(),
    "red_rate_dark": dark["red_rate"].mean(),
    "red_rate_all": player["red_rate"].mean(),
}

# Rate ratio using totals
light_tot_rate = light["redCards"].sum() / light["games"].sum()
dark_tot_rate = dark["redCards"].sum() / dark["games"].sum()
rate_ratio = dark_tot_rate / light_tot_rate if light_tot_rate > 0 else np.nan

# Poisson regression: continuous skin
player["log_games"] = np.log(player["games"])
X = sm.add_constant(player["skin"])
poisson_model = sm.GLM(player["redCards"], X, family=sm.families.Poisson(), offset=player["log_games"])
poisson_res = poisson_model.fit(cov_type="HC0")

# Poisson regression: dark vs light (exclude middle)
player_dl = player[(player["skin"] <= 0.4) | (player["skin"] >= 0.6)].copy()
player_dl["dark"] = (player_dl["skin"] >= 0.6).astype(int)
X_dl = sm.add_constant(player_dl["dark"])
poisson_dl = sm.GLM(player_dl["redCards"], X_dl, family=sm.families.Poisson(), offset=np.log(player_dl["games"]))
poisson_dl_res = poisson_dl.fit(cov_type="HC0")

# Extract results
cont_coef = poisson_res.params["skin"]
cont_se = poisson_res.bse["skin"]
cont_p = poisson_res.pvalues["skin"]

# Rate ratio per 0.1 increase in skin tone
rr_0_1 = float(np.exp(cont_coef * 0.1))

# Dark vs light rate ratio
dl_coef = poisson_dl_res.params["dark"]
dl_p = poisson_dl_res.pvalues["dark"]
rr_dark_light = float(np.exp(dl_coef))

# Overdispersion check
pearson_chi2 = poisson_res.pearson_chi2
pearson_df = poisson_res.df_resid
od_ratio = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

print("SUMMARY")
for k, v in summary.items():
    print(f"{k}: {v}")
print(f"light_total_rate: {light_tot_rate}")
print(f"dark_total_rate: {dark_tot_rate}")
print(f"rate_ratio_dark_light_totals: {rate_ratio}")

print("\nPOISSON CONTINUOUS SKIN")
print(f"coef_skin: {cont_coef}")
print(f"se_skin: {cont_se}")
print(f"p_skin: {cont_p}")
print(f"rr_per_0.1_skin: {rr_0_1}")
print(f"overdispersion_ratio: {od_ratio}")

print("\nPOISSON DARK VS LIGHT")
print(f"coef_dark: {dl_coef}")
print(f"p_dark: {dl_p}")
print(f"rr_dark_vs_light: {rr_dark_light}")

