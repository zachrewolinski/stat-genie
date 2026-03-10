import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF_PATH = "soccer.csv"

df = pd.read_csv(DF_PATH)

# Compute average skin tone rating
skin_cols = ["rater1", "rater2"]

df["skin_avg"] = df[skin_cols].mean(axis=1, skipna=True)

# Keep rows with at least one skin rating
skin_df = df[df["skin_avg"].notna()].copy()

# Basic counts
n_total = len(df)

n_skin = len(skin_df)

# Define skin tone groups
skin_df["skin_extreme"] = np.where(
    skin_df["skin_avg"] <= 0.25, "light",
    np.where(skin_df["skin_avg"] >= 0.75, "dark", "mid")
)

skin_df["skin_binary"] = np.where(
    skin_df["skin_avg"] < 0.5, "light",
    np.where(skin_df["skin_avg"] > 0.5, "dark", "mid")
)

# Red card rate per game
skin_df["red_rate"] = skin_df["redCards"] / skin_df["games"]

# Aggregate by extreme groups
extreme = skin_df[skin_df["skin_extreme"].isin(["light", "dark"])].copy()

extreme_summary = extreme.groupby("skin_extreme").agg(
    rows=("redCards", "size"),
    total_red=("redCards", "sum"),
    total_games=("games", "sum"),
    mean_red_rate=("red_rate", "mean"),
)

# Compute rate ratio (dark vs light) using totals
if set(extreme_summary.index) == {"light", "dark"}:
    rate_light = extreme_summary.loc["light", "total_red"] / extreme_summary.loc["light", "total_games"]
    rate_dark = extreme_summary.loc["dark", "total_red"] / extreme_summary.loc["dark", "total_games"]
    rate_ratio_extreme = rate_dark / rate_light if rate_light > 0 else np.nan
else:
    rate_light = np.nan
    rate_dark = np.nan
    rate_ratio_extreme = np.nan

# Poisson regression with offset (games)
# Use cluster-robust SE by player to account for repeated measures per player.
skin_df["log_games"] = np.log(skin_df["games"])

# Basic model
model1 = smf.glm(
    formula="redCards ~ skin_avg",
    data=skin_df,
    family=sm.families.Poisson(),
    offset=skin_df["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": skin_df["playerShort"]})

# Model with covariates (position, league, height, weight)
# Drop rows with missing covariates for this model
cov_df = skin_df.dropna(subset=["position", "leagueCountry", "height", "weight"]).copy()

cov_model = smf.glm(
    formula="redCards ~ skin_avg + C(position) + C(leagueCountry) + height + weight",
    data=cov_df,
    family=sm.families.Poisson(),
    offset=cov_df["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": cov_df["playerShort"]})

# Extract key stats
coef1 = model1.params["skin_avg"]
se1 = model1.bse["skin_avg"]
p1 = model1.pvalues["skin_avg"]

coef2 = cov_model.params["skin_avg"]
se2 = cov_model.bse["skin_avg"]
p2 = cov_model.pvalues["skin_avg"]

# Rate ratio per 1.0 increase in skin_avg (from very light to very dark)
rr1 = float(np.exp(coef1))
rr1_ci = (float(np.exp(coef1 - 1.96 * se1)), float(np.exp(coef1 + 1.96 * se1)))

rr2 = float(np.exp(coef2))
rr2_ci = (float(np.exp(coef2 - 1.96 * se2)), float(np.exp(coef2 + 1.96 * se2)))

# Rate ratio from light (0.25) to dark (0.75)
rr1_light_dark = float(np.exp(coef1 * 0.5))
rr2_light_dark = float(np.exp(coef2 * 0.5))

# Output results
print("TOTAL_ROWS", n_total)
print("ROWS_WITH_SKIN", n_skin)
print("EXTREME_SUMMARY")
print(extreme_summary)
print("EXTREME_RATE_LIGHT", rate_light)
print("EXTREME_RATE_DARK", rate_dark)
print("EXTREME_RATE_RATIO", rate_ratio_extreme)

print("MODEL1_COEF", coef1)
print("MODEL1_SE", se1)
print("MODEL1_P", p1)
print("MODEL1_RR", rr1)
print("MODEL1_RR_CI", rr1_ci)
print("MODEL1_RR_LIGHT_DARK", rr1_light_dark)

print("MODEL2_COEF", coef2)
print("MODEL2_SE", se2)
print("MODEL2_P", p2)
print("MODEL2_RR", rr2)
print("MODEL2_RR_CI", rr2_ci)
print("MODEL2_RR_LIGHT_DARK", rr2_light_dark)
