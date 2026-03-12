import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# compute mean skin tone
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_mean=skin)

# filter for rows with skin and games>0
clean = df[df["skin_mean"].notna() & (df["games"] > 0)].copy()

# basic counts
n_rows = len(clean)
print("rows_with_skin", n_rows)

# Define light vs dark threshold (<=0.5 light, >0.5 dark)
clean["skin_group"] = np.where(clean["skin_mean"] > 0.5, "dark", "light")

# aggregate rates per group
agg = clean.groupby("skin_group").agg(
    redCards_sum=("redCards", "sum"),
    games_sum=("games", "sum"),
    dyads=("redCards", "size")
)
agg["rate_per_game"] = agg["redCards_sum"] / agg["games_sum"]
agg["rate_per_100"] = agg["rate_per_game"] * 100
print("group_rates")
print(agg)

# Poisson regression with offset log(games)
# response redCards, predictor skin_mean
# add intercept
model_df = clean[["redCards", "games", "skin_mean", "playerShort"]].copy()
model_df["log_games"] = np.log(model_df["games"])

# Fit Poisson GLM with cluster-robust SE by player
X = sm.add_constant(model_df["skin_mean"])
poisson_model = sm.GLM(model_df["redCards"], X, family=sm.families.Poisson(), offset=model_df["log_games"])
poisson_res = poisson_model.fit(cov_type="cluster", cov_kwds={"groups": model_df["playerShort"]})
print("poisson_skin_mean_coef", poisson_res.params["skin_mean"], "se", poisson_res.bse["skin_mean"], "p", poisson_res.pvalues["skin_mean"])
print("poisson_skin_mean_IRR", np.exp(poisson_res.params["skin_mean"]))
print("poisson_skin_mean_CI", np.exp(poisson_res.conf_int().loc["skin_mean"]))

# Poisson regression with skin_group indicator
model_df2 = clean[["redCards", "games", "skin_group", "playerShort"]].copy()
model_df2["log_games"] = np.log(model_df2["games"])
model_df2["dark"] = (model_df2["skin_group"] == "dark").astype(int)
X2 = sm.add_constant(model_df2["dark"])
poisson_model2 = sm.GLM(model_df2["redCards"], X2, family=sm.families.Poisson(), offset=model_df2["log_games"])
poisson_res2 = poisson_model2.fit(cov_type="cluster", cov_kwds={"groups": model_df2["playerShort"]})
print("poisson_dark_coef", poisson_res2.params["dark"], "se", poisson_res2.bse["dark"], "p", poisson_res2.pvalues["dark"])
print("poisson_dark_IRR", np.exp(poisson_res2.params["dark"]))
print("poisson_dark_CI", np.exp(poisson_res2.conf_int().loc["dark"]))

# As robustness: Negative binomial (NB2) to handle overdispersion
nb_model = sm.GLM(model_df["redCards"], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=model_df["log_games"])
nb_res = nb_model.fit(cov_type="cluster", cov_kwds={"groups": model_df["playerShort"]})
print("nb_skin_mean_coef", nb_res.params["skin_mean"], "se", nb_res.bse["skin_mean"], "p", nb_res.pvalues["skin_mean"])
print("nb_skin_mean_IRR", np.exp(nb_res.params["skin_mean"]))
print("nb_skin_mean_CI", np.exp(nb_res.conf_int().loc["skin_mean"]))

# Simple correlation between skin_mean and redCards rate per game at dyad level
clean["red_rate"] = clean["redCards"] / clean["games"]
# use Spearman and Pearson
from scipy import stats
pearson = stats.pearsonr(clean["skin_mean"], clean["red_rate"])
spearman = stats.spearmanr(clean["skin_mean"], clean["red_rate"])
print("pearson_r", pearson)
print("spearman_r", spearman)

# Aggregate at player level to reduce repeated dyads
player = clean.groupby("playerShort").agg(
    skin_mean=("skin_mean", "mean"),
    redCards_sum=("redCards", "sum"),
    games_sum=("games", "sum"),
)
player = player[player["games_sum"] > 0]
player["red_rate"] = player["redCards_sum"] / player["games_sum"]
pearson_p = stats.pearsonr(player["skin_mean"], player["red_rate"])
spearman_p = stats.spearmanr(player["skin_mean"], player["red_rate"])
print("player_level_pearson", pearson_p)
print("player_level_spearman", spearman_p)
