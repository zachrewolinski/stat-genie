import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute mean skin tone across raters (0 to 1 scale)
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin=skin)

# Keep rows with skin tone and valid games
analysis_df = df[df["skin"].notna() & df["games"].notna() & (df["games"] > 0)].copy()

# Binary definitions
analysis_df["skin_dark_50"] = (analysis_df["skin"] > 0.5).astype(int)
analysis_df["skin_dark_extreme"] = np.where(analysis_df["skin"] >= 0.75, 1, np.where(analysis_df["skin"] <= 0.25, 0, np.nan))

# Helper: rate ratio with Wald CI

def rate_ratio_ci(red_dark, games_dark, red_light, games_light, alpha=0.05):
    rate_dark = red_dark / games_dark
    rate_light = red_light / games_light
    rr = rate_dark / rate_light
    # Wald CI for log RR using Poisson counts
    var = (1 / red_dark) + (1 / red_light)
    se = np.sqrt(var)
    z = 1.96
    ci_low = np.exp(np.log(rr) - z * se)
    ci_high = np.exp(np.log(rr) + z * se)
    return rr, ci_low, ci_high

results = {}

for label, group_col in [
    ("threshold_0.5", "skin_dark_50"),
    ("extremes_0.25_0.75", "skin_dark_extreme"),
]:
    sub = analysis_df.copy()
    if group_col == "skin_dark_extreme":
        sub = sub[sub[group_col].notna()].copy()
        sub[group_col] = sub[group_col].astype(int)

    # Aggregated rates
    agg = sub.groupby(group_col).agg(redCards_sum=("redCards", "sum"), games_sum=("games", "sum"), n=("redCards", "size"))
    # Ensure both groups
    if set(agg.index) != {0, 1}:
        results[label] = {"error": "missing group"}
        continue

    red_light = agg.loc[0, "redCards_sum"]
    games_light = agg.loc[0, "games_sum"]
    red_dark = agg.loc[1, "redCards_sum"]
    games_dark = agg.loc[1, "games_sum"]

    rr, ci_low, ci_high = rate_ratio_ci(red_dark, games_dark, red_light, games_light)

    results[label] = {
        "agg": agg.to_dict(),
        "rate_light": red_light / games_light,
        "rate_dark": red_dark / games_dark,
        "rate_ratio": rr,
        "rate_ratio_ci": [ci_low, ci_high],
    }

# Poisson regression with offset(log(games))
# Model 1: continuous skin
sub = analysis_df.copy()
sub = sub[sub["redCards"].notna()]
sub["log_games"] = np.log(sub["games"])

poisson_cont = sm.GLM(sub["redCards"], sm.add_constant(sub[["skin"]]),
                      family=sm.families.Poisson(), offset=sub["log_games"]).fit(cov_type="HC1")

# Model 2: binary threshold
sub2 = analysis_df.copy()
sub2 = sub2[sub2["redCards"].notna()]
sub2["log_games"] = np.log(sub2["games"])
poisson_bin = sm.GLM(sub2["redCards"], sm.add_constant(sub2[["skin_dark_50"]]),
                     family=sm.families.Poisson(), offset=sub2["log_games"]).fit(cov_type="HC1")

results["poisson_continuous"] = {
    "coef_skin": float(poisson_cont.params["skin"]),
    "se_skin": float(poisson_cont.bse["skin"]),
    "pvalue_skin": float(poisson_cont.pvalues["skin"]),
    "rate_ratio_per_unit": float(np.exp(poisson_cont.params["skin"])),
    "n_obs": int(poisson_cont.nobs),
}

results["poisson_binary"] = {
    "coef_dark": float(poisson_bin.params["skin_dark_50"]),
    "se_dark": float(poisson_bin.bse["skin_dark_50"]),
    "pvalue_dark": float(poisson_bin.pvalues["skin_dark_50"]),
    "rate_ratio_dark_vs_light": float(np.exp(poisson_bin.params["skin_dark_50"])),
    "n_obs": int(poisson_bin.nobs),
}

print(json.dumps(results, indent=2))
