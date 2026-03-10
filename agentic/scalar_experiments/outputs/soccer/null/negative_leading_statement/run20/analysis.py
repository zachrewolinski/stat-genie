import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

# Load dataset
# Use low_memory=False to ensure proper dtypes
_df = pd.read_csv(DATA_PATH, low_memory=False)

# Compute skin tone (mean of raters when available)
_df["skin"] = _df[["rater1", "rater2"]].mean(axis=1, skipna=True)

# Keep rows with at least one skin rating and valid games
_df = _df[_df["skin"].notna() & _df["games"].notna() & (_df["games"] > 0)]

# Basic counts
n_total = len(_df)

# Define light/dark groups based on 5-point scale (0,0.25,0.5,0.75,1)
light = _df[_df["skin"] <= 0.25].copy()
dark = _df[_df["skin"] >= 0.75].copy()

# Aggregate red cards and games for light/dark

def summarize_group(gdf):
    total_red = gdf["redCards"].sum()
    total_games = gdf["games"].sum()
    rate = total_red / total_games if total_games > 0 else np.nan
    return total_red, total_games, rate

light_red, light_games, light_rate = summarize_group(light)
dark_red, dark_games, dark_rate = summarize_group(dark)

# Poisson regression: redCards ~ skin with offset log(games)
# Use GLM Poisson with log link; offset accounts for exposure (games)
# Add constant
X = sm.add_constant(_df[["skin"]])
model = sm.GLM(_df["redCards"], X, family=sm.families.Poisson(), offset=np.log(_df["games"]))
res = model.fit()

# Poisson regression: redCards ~ dark (binary), restricted to light/dark for clear comparison
if len(light) > 0 and len(dark) > 0:
    ld = pd.concat([light.assign(dark=0), dark.assign(dark=1)], axis=0)
    X_ld = sm.add_constant(ld[["dark"]])
    model_ld = sm.GLM(ld["redCards"], X_ld, family=sm.families.Poisson(), offset=np.log(ld["games"]))
    res_ld = model_ld.fit()
else:
    res_ld = None

# Also run a logistic regression for any red card occurrence per dyad (redCards>0) as sensitivity
_df["red_any"] = (_df["redCards"] > 0).astype(int)
X_log = sm.add_constant(_df[["skin"]])
logit = sm.GLM(_df["red_any"], X_log, family=sm.families.Binomial())
res_log = logit.fit()

# Print summary stats
print("Rows with skin rating:", n_total)
print("Light group rows:", len(light), "Dark group rows:", len(dark))
print("Light total red cards:", light_red, "Light total games:", light_games, "Light rate per game:", light_rate)
print("Dark total red cards:", dark_red, "Dark total games:", dark_games, "Dark rate per game:", dark_rate)
if light_rate > 0:
    print("Rate ratio dark/light:", dark_rate / light_rate)

print("\nPoisson (redCards ~ skin, offset log(games))")
print(res.summary())

if res_ld is not None:
    print("\nPoisson (redCards ~ dark, offset log(games), light vs dark subset)")
    print(res_ld.summary())

print("\nLogit (red_any ~ skin)")
print(res_log.summary())

# Save key results for downstream use
results = {
    "n_total": int(n_total),
    "light_n": int(len(light)),
    "dark_n": int(len(dark)),
    "light_red": float(light_red),
    "dark_red": float(dark_red),
    "light_games": float(light_games),
    "dark_games": float(dark_games),
    "light_rate": float(light_rate),
    "dark_rate": float(dark_rate),
    "rate_ratio_dark_light": float(dark_rate / light_rate) if light_rate > 0 else np.nan,
    "poisson_skin_coef": float(res.params["skin"]),
    "poisson_skin_p": float(res.pvalues["skin"]),
    "poisson_skin_ci_low": float(res.conf_int().loc["skin", 0]),
    "poisson_skin_ci_high": float(res.conf_int().loc["skin", 1]),
    "logit_skin_coef": float(res_log.params["skin"]),
    "logit_skin_p": float(res_log.pvalues["skin"]),
    "logit_skin_ci_low": float(res_log.conf_int().loc["skin", 0]),
    "logit_skin_ci_high": float(res_log.conf_int().loc["skin", 1]),
}

if res_ld is not None:
    results.update({
        "poisson_dark_coef": float(res_ld.params["dark"]),
        "poisson_dark_p": float(res_ld.pvalues["dark"]),
        "poisson_dark_ci_low": float(res_ld.conf_int().loc["dark", 0]),
        "poisson_dark_ci_high": float(res_ld.conf_int().loc["dark", 1]),
    })

pd.DataFrame([results]).to_csv("analysis_results.csv", index=False)
