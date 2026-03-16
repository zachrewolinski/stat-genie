import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = "soccer.csv"

df = pd.read_csv(csv_path)

# Compute mean skin tone from raters when available
skin = df[["rater1", "rater2"]].mean(axis=1)

df = df.assign(skin=skin)

# Aggregate to player level to avoid multiple dyads per player
# Use playerShort as unique player id
player_cols = ["playerShort", "skin"]

player_df = (
    df
    .dropna(subset=["skin", "games", "redCards"])
    .groupby(player_cols, as_index=False)
    .agg(total_games=("games", "sum"), total_red=("redCards", "sum"))
)

# Filter players with at least 1 game
player_df = player_df[player_df["total_games"] > 0].copy()

# Define light and dark skin groups using 5-point scale normalized to [0,1]
# Light: <=0.25 (very light/light), Dark: >=0.75 (dark/very dark)
light_threshold = 0.25
dark_threshold = 0.75

player_df["skin_group"] = np.where(
    player_df["skin"] <= light_threshold, "light",
    np.where(player_df["skin"] >= dark_threshold, "dark", "mid")
)

# Compute rate per game
player_df["red_rate"] = player_df["total_red"] / player_df["total_games"]

light = player_df[player_df["skin_group"] == "light"]
dark = player_df[player_df["skin_group"] == "dark"]

# Summary stats
summary = {
    "n_players_total": len(player_df),
    "n_light": len(light),
    "n_dark": len(dark),
    "mean_rate_light": light["red_rate"].mean(),
    "mean_rate_dark": dark["red_rate"].mean(),
    "median_rate_light": light["red_rate"].median(),
    "median_rate_dark": dark["red_rate"].median(),
}

# Welch t-test on rates
if len(light) > 1 and len(dark) > 1:
    t_stat, t_p = stats.ttest_ind(light["red_rate"], dark["red_rate"], equal_var=False)
else:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U test (nonparametric)
if len(light) > 1 and len(dark) > 1:
    u_stat, u_p = stats.mannwhitneyu(light["red_rate"], dark["red_rate"], alternative="two-sided")
else:
    u_stat, u_p = np.nan, np.nan

# Poisson regression with offset log(games)
# Binary predictor: dark=1, light=0; drop mid
reg_df = player_df[player_df["skin_group"].isin(["light", "dark"])].copy()
reg_df["dark"] = (reg_df["skin_group"] == "dark").astype(int)

# Use log(total_games) as offset
reg_df["log_games"] = np.log(reg_df["total_games"])

# Add constant
X = sm.add_constant(reg_df["dark"], has_constant="add")

# Poisson GLM
poisson_model = sm.GLM(reg_df["total_red"], X, family=sm.families.Poisson(), offset=reg_df["log_games"])
poisson_res = poisson_model.fit()

# Robust SE (HC3)
poisson_res_robust = poisson_model.fit(cov_type="HC3")

# Incidence Rate Ratio for dark vs light
coef_dark = poisson_res.params["dark"]
se_dark = poisson_res_robust.bse["dark"]

irr = np.exp(coef_dark)
ci_low = np.exp(coef_dark - 1.96 * se_dark)
ci_high = np.exp(coef_dark + 1.96 * se_dark)

# p-value for dark coefficient (robust)
p_dark = poisson_res_robust.pvalues["dark"]

print("SUMMARY", summary)
print("TTEST", t_stat, t_p)
print("MANNWHITNEY", u_stat, u_p)
print("POISSON_COEF_DARK", coef_dark)
print("POISSON_IRR", irr)
print("POISSON_IRR_CI", ci_low, ci_high)
print("POISSON_P", p_dark)

# Sensitivity: treat skin as continuous (mean skin) across all players
# Poisson GLM with continuous skin, using all players with skin
reg_df2 = player_df.copy()
reg_df2["log_games"] = np.log(reg_df2["total_games"])
X2 = sm.add_constant(reg_df2["skin"], has_constant="add")
poisson_model2 = sm.GLM(reg_df2["total_red"], X2, family=sm.families.Poisson(), offset=reg_df2["log_games"])
poisson_res2 = poisson_model2.fit()
poisson_res2_robust = poisson_model2.fit(cov_type="HC3")
coef_skin = poisson_res2.params["skin"]
se_skin = poisson_res2_robust.bse["skin"]

irr_skin = np.exp(coef_skin)
ci_low2 = np.exp(coef_skin - 1.96 * se_skin)
ci_high2 = np.exp(coef_skin + 1.96 * se_skin)

p_skin = poisson_res2_robust.pvalues["skin"]

print("POISSON_CONT_COEF", coef_skin)
print("POISSON_CONT_IRR", irr_skin)
print("POISSON_CONT_CI", ci_low2, ci_high2)
print("POISSON_CONT_P", p_skin)
