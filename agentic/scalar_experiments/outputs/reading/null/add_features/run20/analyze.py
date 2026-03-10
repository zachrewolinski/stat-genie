import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Prefer dyslexia_bin if present; otherwise dyslexia>0
if "dyslexia_bin" in df.columns:
    dys = df["dyslexia_bin"].copy()
else:
    dys = (df["dyslexia"] > 0).astype(int)

# Filter to individuals with dyslexia
mask = dys == 1
sub = df.loc[mask].copy()

# Ensure required columns exist
required_cols = ["speed", "reader_view", "uuid", "page_id"]
missing = [c for c in required_cols if c not in sub.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop missing/invalid values
sub = sub.dropna(subset=["speed", "reader_view", "uuid", "page_id"]).copy()
# Keep non-negative speeds
sub = sub[sub["speed"] > 0].copy()

# Descriptive stats
sub["reader_view"] = sub["reader_view"].astype(int)

desc = sub.groupby("reader_view")["speed"].agg([
    "count", "mean", "median", "std"
]).reset_index()

# Log-transform for modeling
sub["log_speed"] = np.log(sub["speed"])

# Welch t-test on log speed
rv0 = sub.loc[sub["reader_view"] == 0, "log_speed"]
rv1 = sub.loc[sub["reader_view"] == 1, "log_speed"]

t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Mann-Whitney U on raw speed (robust)
try:
    u_stat, u_p = stats.mannwhitneyu(
        sub.loc[sub["reader_view"] == 1, "speed"],
        sub.loc[sub["reader_view"] == 0, "speed"],
        alternative="two-sided"
    )
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Mixed effects model: log_speed ~ reader_view + page_id, random intercept per uuid
# Use category for page_id
sub["page_id"] = sub["page_id"].astype("category")

model = smf.mixedlm("log_speed ~ reader_view + C(page_id)", sub, groups=sub["uuid"])
try:
    fit = model.fit(reml=False, method="lbfgs")
    coef = fit.params.get("reader_view", np.nan)
    pval = fit.pvalues.get("reader_view", np.nan)
except Exception as e:
    fit = None
    coef = np.nan
    pval = np.nan

# Convert log coefficient to percent change
if np.isfinite(coef):
    pct_change = (np.exp(coef) - 1) * 100
else:
    pct_change = np.nan

result = {
    "n_rows": int(len(sub)),
    "n_uuid": int(sub["uuid"].nunique()),
    "desc": desc.to_dict(orient="records"),
    "ttest_log": {"t": float(t_stat), "p": float(t_p)},
    "mannwhitney": {"u": float(u_stat) if np.isfinite(u_stat) else None, "p": float(u_p) if np.isfinite(u_p) else None},
    "mixedlm": {
        "coef_log": float(coef) if np.isfinite(coef) else None,
        "p": float(pval) if np.isfinite(pval) else None,
        "pct_change": float(pct_change) if np.isfinite(pct_change) else None,
    },
}

print(json.dumps(result, indent=2))
