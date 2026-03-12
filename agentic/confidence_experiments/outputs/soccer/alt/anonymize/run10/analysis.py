import json

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("soccer.csv")

# Compute average skin tone rating (normalized 0-1 from two raters)
df["skin_avg"] = df[["feature18", "feature19"]].mean(axis=1)

# Filter for available skin ratings and positive games
df_skin = df.dropna(subset=["skin_avg"]).copy()
df_skin = df_skin[df_skin["feature9"] > 0].copy()

# Map to 1-5 scale for clearer thresholds
df_skin["skin_avg_1_5"] = df_skin["skin_avg"] * 4 + 1

# Define dark/light groups using 1-5 scale: light <=2, dark >=4
df_skin["skin_group"] = np.where(
    df_skin["skin_avg_1_5"] >= 4,
    "dark",
    np.where(df_skin["skin_avg_1_5"] <= 2, "light", "mid"),
)

# Aggregate rates for dark vs light
df_dl = df_skin[df_skin["skin_group"].isin(["dark", "light"])].copy()


def group_rate_stats(group_df):
    red = group_df["feature16"].sum()
    games = group_df["feature9"].sum()
    rate = red / games if games > 0 else np.nan
    return red, games, rate


red_dark, games_dark, rate_dark = group_rate_stats(df_dl[df_dl["skin_group"] == "dark"])
red_light, games_light, rate_light = group_rate_stats(df_dl[df_dl["skin_group"] == "light"])

# Rate ratio + approx CI
if red_dark > 0 and red_light > 0:
    log_rr = np.log(rate_dark / rate_light)
    se_log_rr = np.sqrt(1 / red_dark + 1 / red_light)
    z = log_rr / se_log_rr
    p_rr = 2 * (1 - stats.norm.cdf(abs(z)))
    rr_ci_low = np.exp(log_rr - 1.96 * se_log_rr)
    rr_ci_high = np.exp(log_rr + 1.96 * se_log_rr)
else:
    log_rr = np.nan
    se_log_rr = np.nan
    p_rr = np.nan
    rr_ci_low = np.nan
    rr_ci_high = np.nan

# Mann-Whitney U test on per-game rate (dyad level)
df_dl["red_rate"] = df_dl["feature16"] / df_dl["feature9"]
rates_dark = df_dl.loc[df_dl["skin_group"] == "dark", "red_rate"]
rates_light = df_dl.loc[df_dl["skin_group"] == "light", "red_rate"]

try:
    mw_stat, mw_p = stats.mannwhitneyu(rates_dark, rates_light, alternative="two-sided")
except ValueError:
    mw_stat, mw_p = np.nan, np.nan

# Poisson regression with offset (clustered SE by player)
df_skin["log_games"] = np.log(df_skin["feature9"])

model_cont = smf.glm(
    formula="feature16 ~ skin_avg",
    data=df_skin,
    family=sm.families.Poisson(),
    offset=df_skin["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": df_skin["feature1"]})

beta_cont = model_cont.params["skin_avg"]
p_cont = model_cont.pvalues["skin_avg"]

# Effect from light(0.25) to dark(0.75) = 0.5 increase
rr_cont_05 = float(np.exp(beta_cont * 0.5))

# Poisson regression dark vs light only
df_dl["dark"] = (df_dl["skin_group"] == "dark").astype(int)
df_dl["log_games"] = np.log(df_dl["feature9"])

model_dl = smf.glm(
    formula="feature16 ~ dark",
    data=df_dl,
    family=sm.families.Poisson(),
    offset=df_dl["log_games"],
).fit(cov_type="cluster", cov_kwds={"groups": df_dl["feature1"]})

beta_dl = model_dl.params["dark"]
p_dl = model_dl.pvalues["dark"]
rr_dl = float(np.exp(beta_dl))

# Summaries for explanation
summary = {
    "n_total": int(len(df)),
    "n_skin": int(len(df_skin)),
    "n_dark": int((df_dl["skin_group"] == "dark").sum()),
    "n_light": int((df_dl["skin_group"] == "light").sum()),
    "red_dark": float(red_dark),
    "red_light": float(red_light),
    "games_dark": float(games_dark),
    "games_light": float(games_light),
    "rate_dark": float(rate_dark),
    "rate_light": float(rate_light),
    "rr_rate": float(rate_dark / rate_light) if rate_light > 0 else np.nan,
    "rr_ci_low": float(rr_ci_low),
    "rr_ci_high": float(rr_ci_high),
    "p_rr": float(p_rr),
    "mw_p": float(mw_p),
    "poisson_beta_cont": float(beta_cont),
    "poisson_p_cont": float(p_cont),
    "rr_cont_05": rr_cont_05,
    "poisson_beta_dl": float(beta_dl),
    "poisson_p_dl": float(p_dl),
    "rr_dl": rr_dl,
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
