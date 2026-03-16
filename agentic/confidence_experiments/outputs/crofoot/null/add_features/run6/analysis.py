import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy import stats

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Core variables
# Relative group size (focal - other)
df["size_diff"] = df["n_focal"] - df["n_other"]
# Relative location: positive means focal closer to its own home center
# (other is farther from its own center than focal is from its center)
df["loc_diff"] = df["dist_other"] - df["dist_focal"]
# Binary home advantage (focal closer to its center than other is to its center)
df["home_adv"] = (df["dist_focal"] < df["dist_other"]).astype(int)

# Prepare model data
X = df[["size_diff", "loc_diff"]].copy()
X = sm.add_constant(X)
y = df["win"].astype(int)

results = {}

# Logistic regression with size_diff and loc_diff
try:
    model = sm.Logit(y, X).fit(disp=False)
    results["logit_full"] = model
except PerfectSeparationError:
    results["logit_full"] = None

# Logistic regression with size_diff only
try:
    model_size = sm.Logit(y, sm.add_constant(df[["size_diff"]])).fit(disp=False)
    results["logit_size"] = model_size
except PerfectSeparationError:
    results["logit_size"] = None

# Logistic regression with loc_diff only
try:
    model_loc = sm.Logit(y, sm.add_constant(df[["loc_diff"]])).fit(disp=False)
    results["logit_loc"] = model_loc
except PerfectSeparationError:
    results["logit_loc"] = None

# Descriptive stats
win_rate_overall = df["win"].mean()

# Win rate when focal has size advantage / disadvantage / equal
size_adv = df["size_diff"] > 0
size_disadv = df["size_diff"] < 0
size_equal = df["size_diff"] == 0

win_rate_size_adv = df.loc[size_adv, "win"].mean() if size_adv.any() else np.nan
win_rate_size_disadv = df.loc[size_disadv, "win"].mean() if size_disadv.any() else np.nan
win_rate_size_equal = df.loc[size_equal, "win"].mean() if size_equal.any() else np.nan

# Win rate by home advantage
win_rate_home_adv = df.loc[df["home_adv"] == 1, "win"].mean()
win_rate_home_disadv = df.loc[df["home_adv"] == 0, "win"].mean()

# Mean differences and Welch t-tests
win_mask = df["win"] == 1
lose_mask = df["win"] == 0

size_win = df.loc[win_mask, "size_diff"]
size_lose = df.loc[lose_mask, "size_diff"]
loc_win = df.loc[win_mask, "loc_diff"]
loc_lose = df.loc[lose_mask, "loc_diff"]

ttest_size = stats.ttest_ind(size_win, size_lose, equal_var=False)
ttest_loc = stats.ttest_ind(loc_win, loc_lose, equal_var=False)

# Summaries
summary = {
    "n": int(df.shape[0]),
    "win_rate_overall": float(win_rate_overall),
    "win_rate_size_adv": float(win_rate_size_adv),
    "win_rate_size_disadv": float(win_rate_size_disadv),
    "win_rate_size_equal": float(win_rate_size_equal),
    "win_rate_home_adv": float(win_rate_home_adv),
    "win_rate_home_disadv": float(win_rate_home_disadv),
    "mean_size_diff_win": float(size_win.mean()),
    "mean_size_diff_lose": float(size_lose.mean()),
    "mean_loc_diff_win": float(loc_win.mean()),
    "mean_loc_diff_lose": float(loc_lose.mean()),
    "ttest_size": {
        "stat": float(ttest_size.statistic),
        "pvalue": float(ttest_size.pvalue),
    },
    "ttest_loc": {
        "stat": float(ttest_loc.statistic),
        "pvalue": float(ttest_loc.pvalue),
    },
}

# Extract coefficients, p-values, and odds ratios
model_outputs = {}
for key, model in results.items():
    if model is None:
        model_outputs[key] = None
        continue
    params = model.params
    pvalues = model.pvalues
    conf = model.conf_int()
    # odds ratios for interpretability
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)
    model_outputs[key] = {
        "params": params.to_dict(),
        "pvalues": pvalues.to_dict(),
        "odds_ratios": odds_ratios.to_dict(),
        "conf_int": conf.to_dict(),
        "conf_int_or": conf_or.to_dict(),
        "pseudo_r2": float(model.prsquared),
        "aic": float(model.aic),
    }

# Save a compact JSON for inspection
output = {
    "summary": summary,
    "models": model_outputs,
}

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
