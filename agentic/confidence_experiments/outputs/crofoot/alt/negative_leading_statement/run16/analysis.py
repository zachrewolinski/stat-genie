import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv("crofoot.csv")

# Derived variables
_df["rel_size"] = _df["n_focal"] - _df["n_other"]
_df["loc_adv"] = _df["dist_other"] - _df["dist_focal"]  # positive => contest closer to focal group's center
_df["loc_adv_100"] = _df["loc_adv"] / 100.0

# Basic stats
n = len(_df)
win_rate = _df["win"].mean()

# Group comparisons
win_group = _df[_df["win"] == 1]
lose_group = _df[_df["win"] == 0]

mean_rel_win = win_group["rel_size"].mean()
mean_rel_lose = lose_group["rel_size"].mean()
mean_loc_win = win_group["loc_adv"].mean()
mean_loc_lose = lose_group["loc_adv"].mean()

# t-tests (Welch)
rel_t = stats.ttest_ind(win_group["rel_size"], lose_group["rel_size"], equal_var=False)
loc_t = stats.ttest_ind(win_group["loc_adv"], lose_group["loc_adv"], equal_var=False)

# Logistic regression: win ~ rel_size + loc_adv_100
model = smf.glm("win ~ rel_size + loc_adv_100", data=_df, family=sm.families.Binomial())
res = model.fit()

# Cluster-robust SE by dyad (to account for repeated dyads)
try:
    res_cluster = model.fit(cov_type="cluster", cov_kwds={"groups": _df["dyad"]})
except Exception:
    res_cluster = None

# Single-predictor models
model_rel = smf.glm("win ~ rel_size", data=_df, family=sm.families.Binomial()).fit()
model_loc = smf.glm("win ~ loc_adv_100", data=_df, family=sm.families.Binomial()).fit()

# Summaries
out = {
    "n": int(n),
    "win_rate": float(win_rate),
    "mean_rel_size_win": float(mean_rel_win),
    "mean_rel_size_lose": float(mean_rel_lose),
    "mean_loc_adv_win": float(mean_loc_win),
    "mean_loc_adv_lose": float(mean_loc_lose),
    "t_rel_size": {"stat": float(rel_t.statistic), "pvalue": float(rel_t.pvalue)},
    "t_loc_adv": {"stat": float(loc_t.statistic), "pvalue": float(loc_t.pvalue)},
    "logit_main": {
        "params": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "conf_int": res.conf_int().rename(columns={0: "low", 1: "high"}).to_dict(orient="index"),
        "aic": float(res.aic),
    },
    "logit_rel_only": {
        "params": model_rel.params.to_dict(),
        "pvalues": model_rel.pvalues.to_dict(),
        "aic": float(model_rel.aic),
    },
    "logit_loc_only": {
        "params": model_loc.params.to_dict(),
        "pvalues": model_loc.pvalues.to_dict(),
        "aic": float(model_loc.aic),
    },
}

if res_cluster is not None:
    out["logit_main_cluster"] = {
        "params": res_cluster.params.to_dict(),
        "pvalues": res_cluster.pvalues.to_dict(),
        "conf_int": res_cluster.conf_int().rename(columns={0: "low", 1: "high"}).to_dict(orient="index"),
    }

print(json.dumps(out, indent=2))
