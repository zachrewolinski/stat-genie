import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Compute mean skin tone from two raters; keep rows with at least one rating
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Keep valid observations
_df = df[(df["skin_tone"].notna()) & (df["games"] > 0)].copy()

# Define light vs dark using ends of the 5-point scale (0, 0.25, 0.5, 0.75, 1.0)
_df["light"] = _df["skin_tone"] <= 0.25
_df["dark"] = _df["skin_tone"] >= 0.75

# Summary stats for light vs dark
light = _df[_df["light"]]
dark = _df[_df["dark"]]

# Red cards per game
light_rate = (light["redCards"] / light["games"]).mean()
dark_rate = (dark["redCards"] / dark["games"]).mean()

# Welch t-test on red cards per game
light_vals = (light["redCards"] / light["games"]).values
dark_vals = (dark["redCards"] / dark["games"]).values

ttest = stats.ttest_ind(dark_vals, light_vals, equal_var=False, nan_policy="omit")

# Poisson regression with exposure (games)
_df["log_games"] = np.log(_df["games"])  # offset

# Model 1: continuous skin tone
model_cont = smf.glm(
    formula="redCards ~ skin_tone",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_games"],
).fit(cov_type="HC0")

# Model 2: dark indicator (vs others)
model_dark = smf.glm(
    formula="redCards ~ dark",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_games"],
).fit(cov_type="HC0")

# Model 3: controls (leagueCountry, position)
model_ctrl = smf.glm(
    formula="redCards ~ skin_tone + C(leagueCountry) + C(position)",
    data=_df,
    family=sm.families.Poisson(),
    offset=_df["log_games"],
).fit(cov_type="HC0")

# Dispersion check
pearson_chi2 = model_cont.pearson_chi2
pearson_df = model_cont.df_resid
pearson_dispersion = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

results = {
    "n_total": int(len(df)),
    "n_skin": int(len(_df)),
    "n_light": int(len(light)),
    "n_dark": int(len(dark)),
    "light_rate": float(light_rate),
    "dark_rate": float(dark_rate),
    "ttest_stat": float(ttest.statistic),
    "ttest_p": float(ttest.pvalue),
    "model_cont_coef": float(model_cont.params.get("skin_tone", np.nan)),
    "model_cont_p": float(model_cont.pvalues.get("skin_tone", np.nan)),
    "model_cont_rr": float(np.exp(model_cont.params.get("skin_tone", np.nan))),
    "model_dark_coef": float(model_dark.params.get("dark[T.True]", np.nan)),
    "model_dark_p": float(model_dark.pvalues.get("dark[T.True]", np.nan)),
    "model_dark_rr": float(np.exp(model_dark.params.get("dark[T.True]", np.nan))),
    "model_ctrl_coef": float(model_ctrl.params.get("skin_tone", np.nan)),
    "model_ctrl_p": float(model_ctrl.pvalues.get("skin_tone", np.nan)),
    "model_ctrl_rr": float(np.exp(model_ctrl.params.get("skin_tone", np.nan))),
    "pearson_dispersion": float(pearson_dispersion),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
