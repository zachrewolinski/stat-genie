import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Ensure expected columns
print("rows", len(df))
print("columns", df.columns.tolist())

# Identify dyslexia participants
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
elif "dyslexia" in df.columns:
    dys_df = df[df["dyslexia"].fillna(0) > 0].copy()
else:
    raise ValueError("No dyslexia indicator column found")

print("dyslexia rows", len(dys_df))

# Remove non-positive speed if any
speed = dys_df["speed"].astype(float)

dys_df = dys_df[np.isfinite(speed)].copy()

# Basic group stats
stats_by_group = dys_df.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"])
print("speed stats by reader_view")
print(stats_by_group)

# Two-sample Welch t-test
rv1 = dys_df.loc[dys_df["reader_view"] == 1, "speed"].astype(float)
rv0 = dys_df.loc[dys_df["reader_view"] == 0, "speed"].astype(float)

t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
print("Welch t-test", t_stat, p_val)

# Effect size (Cohen's d using pooled std for groups)
mean1, mean0 = rv1.mean(), rv0.mean()
std1, std0 = rv1.std(ddof=1), rv0.std(ddof=1)
# pooled SD with unequal n
n1, n0 = rv1.size, rv0.size
pooled_sd = np.sqrt(((n1 - 1) * std1**2 + (n0 - 1) * std0**2) / (n1 + n0 - 2)) if (n1+n0-2) > 0 else np.nan
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan
print("cohen d", cohen_d)

# Regression with log speed to reduce skew; cluster-robust SE by uuid if available
# Add small constant to avoid log(0)
log_speed = np.log(dys_df["speed"].astype(float).clip(lower=1e-6))
dys_df = dys_df.assign(log_speed=log_speed)

formula = "log_speed ~ reader_view"

if "page_id" in dys_df.columns:
    formula += " + C(page_id)"

# Control for language and device if available (avoid overfitting)
if "device" in dys_df.columns:
    formula += " + C(device)"
if "language" in dys_df.columns:
    formula += " + C(language)"

# Drop rows with missing data in any model column to keep groups aligned
model_cols = ["log_speed", "reader_view"]
if "page_id" in dys_df.columns:
    model_cols.append("page_id")
if "device" in dys_df.columns:
    model_cols.append("device")
if "language" in dys_df.columns:
    model_cols.append("language")
if "uuid" in dys_df.columns:
    model_cols.append("uuid")

model_df = dys_df.dropna(subset=model_cols).copy()

model = smf.ols(formula, data=model_df)
cov_kwds = {"groups": model_df["uuid"]} if "uuid" in model_df.columns else None
res = model.fit(cov_type="cluster" if cov_kwds else "nonrobust", cov_kwds=cov_kwds)
print(res.summary())

# Extract effect
coef = res.params.get("reader_view", np.nan)
se = res.bse.get("reader_view", np.nan)
ci_low, ci_high = res.conf_int().loc["reader_view"].tolist() if "reader_view" in res.params else (np.nan, np.nan)

# Convert log effect to percent change
pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan
ci_low_pct = (np.exp(ci_low) - 1) * 100 if np.isfinite(ci_low) else np.nan
ci_high_pct = (np.exp(ci_high) - 1) * 100 if np.isfinite(ci_high) else np.nan

print("log-speed coef", coef, "SE", se)
print("percent change", pct_change, "CI", (ci_low_pct, ci_high_pct))

# Save summary to json for later use
summary = {
    "n_total": int(len(dys_df)),
    "n_rv1": int(n1),
    "n_rv0": int(n0),
    "mean_rv1": float(mean1),
    "mean_rv0": float(mean0),
    "median_rv1": float(rv1.median()),
    "median_rv0": float(rv0.median()),
    "welch_t": float(t_stat),
    "welch_p": float(p_val),
    "cohen_d": float(cohen_d),
    "log_coef": float(coef),
    "log_ci_low": float(ci_low),
    "log_ci_high": float(ci_high),
    "pct_change": float(pct_change),
    "pct_ci_low": float(ci_low_pct),
    "pct_ci_high": float(ci_high_pct),
    "reg_p": float(res.pvalues.get("reader_view", np.nan)),
    "reg_model_df": int(res.df_model),
    "reg_df_resid": int(res.df_resid),
    "reg_n": int(res.nobs),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("saved analysis_summary.json")
