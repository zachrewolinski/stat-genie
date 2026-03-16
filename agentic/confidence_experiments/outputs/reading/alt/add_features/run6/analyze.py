import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Focus on individuals with dyslexia
# Use dyslexia_bin if present; fallback to dyslexia > 0
if "dyslexia_bin" in df.columns:
    dys = df[df["dyslexia_bin"] == 1].copy()
else:
    dys = df[df["dyslexia"] > 0].copy()

# Keep relevant rows
cols_needed = ["reader_view", "speed", "num_words", "page_id", "device", "age", "gender", "english_native", "retake_trial"]
for c in cols_needed:
    if c not in dys.columns:
        pass

# Basic cleaning: drop missing speed/reader_view, keep positive speeds
dys = dys.dropna(subset=["reader_view", "speed"]).copy()
dys = dys[dys["speed"] > 0].copy()

# Group summaries
summary = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Mann-Whitney U test (nonparametric)
rv0 = dys[dys["reader_view"] == 0]["speed"].values
rv1 = dys[dys["reader_view"] == 1]["speed"].values

mw = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")

# Cliff's delta
# Compute effect size via pairwise comparisons
# Use efficient computation by ranks
import math

def cliffs_delta(x, y):
    # x: treatment (rv1), y: control (rv0)
    # Based on rank sums
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    # If either empty, return nan
    if nx == 0 or ny == 0:
        return np.nan
    # compute delta using scipy.stats.rankdata
    all_vals = np.concatenate([x, y])
    ranks = stats.rankdata(all_vals)
    rx = ranks[:nx].sum()
    # U statistic for x
    Ux = rx - nx*(nx+1)/2
    delta = (2*Ux)/(nx*ny) - 1
    return delta

cd = cliffs_delta(rv1, rv0)

# Welch's t-test on log speed
log_speed = np.log(dys["speed"])
dys["log_speed"] = log_speed
rv0_log = dys[dys["reader_view"] == 0]["log_speed"].values
rv1_log = dys[dys["reader_view"] == 1]["log_speed"].values

ttest = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")

# Cohen's d on log speed

def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)

cd_log = cohens_d(rv1_log, rv0_log)

# OLS regression on log speed with covariates (if columns exist)
# Build formula with available columns
covariates = []
for c in ["num_words", "page_id", "device", "age", "gender", "english_native", "retake_trial"]:
    if c in dys.columns:
        covariates.append(c)

# Create formula, treat categorical with C()
terms = []
for c in covariates:
    if dys[c].dtype == object or str(dys[c].dtype).startswith("category"):
        terms.append(f"C({c})")
    else:
        terms.append(c)

formula = "log_speed ~ reader_view"
if terms:
    formula += " + " + " + ".join(terms)

reg = smf.ols(formula, data=dys).fit()

# Extract coefficient for reader_view
coef = reg.params.get("reader_view", np.nan)
pval = reg.pvalues.get("reader_view", np.nan)

results = {
    "n_total": int(dys.shape[0]),
    "n_reader_view_1": int((dys["reader_view"] == 1).sum()),
    "n_reader_view_0": int((dys["reader_view"] == 0).sum()),
    "speed_summary": summary.to_dict(orient="records"),
    "mannwhitney_u": {"stat": float(mw.statistic), "p": float(mw.pvalue)},
    "cliffs_delta": float(cd),
    "welch_ttest_log": {"stat": float(ttest.statistic), "p": float(ttest.pvalue)},
    "cohens_d_log": float(cd_log),
    "regression": {"formula": formula, "coef_reader_view": float(coef), "p_reader_view": float(pval), "r2": float(reg.rsquared)}
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
