import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Define dyslexia subset
# Use dyslexia_bin if present; fall back to dyslexia > 0
if "dyslexia_bin" in df.columns:
    dyslexia_mask = df["dyslexia_bin"] == 1
else:
    dyslexia_mask = df["dyslexia"] > 0

# Keep required columns
cols = [
    "uuid",
    "reader_view",
    "speed",
    "page_id",
    "num_words",
    "Flesch_Kincaid",
    "device",
    "age",
    "gender",
    "english_native",
]
cols = [c for c in cols if c in df.columns]

sub = df.loc[dyslexia_mask, cols].copy()

# Drop missing critical values
sub = sub.dropna(subset=["reader_view", "speed"])

# Ensure reader_view is binary
sub = sub[sub["reader_view"].isin([0, 1])]

# Log-transform speed to reduce skew
sub["log_speed"] = np.log1p(sub["speed"])

# Descriptive stats
summary = (
    sub.groupby("reader_view")["speed"]
    .agg(["count", "mean", "median", "std"]) 
    .rename(index={0: "No Reader View", 1: "Reader View"})
)

# Difference in means
mean_diff = summary.loc["Reader View", "mean"] - summary.loc["No Reader View", "mean"]
median_diff = summary.loc["Reader View", "median"] - summary.loc["No Reader View", "median"]

# Welch t-test on log speed
rv = sub.loc[sub["reader_view"] == 1, "log_speed"]
no_rv = sub.loc[sub["reader_view"] == 0, "log_speed"]

t_stat, p_val = stats.ttest_ind(rv, no_rv, equal_var=False, nan_policy="omit")

# Mann-Whitney U test on speed
try:
    u_stat, u_p = stats.mannwhitneyu(rv, no_rv, alternative="two-sided")
except Exception:
    u_stat, u_p = np.nan, np.nan

# Regression with cluster-robust SE by participant
# Keep model modest to avoid overfitting
formula = "log_speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid + C(device) + age"

model_data = sub.dropna(subset=["page_id", "num_words", "Flesch_Kincaid", "device", "age", "log_speed"])

if model_data["uuid"].nunique() > 1 and model_data.shape[0] > 10:
    model = smf.ols(formula, data=model_data).fit(
        cov_type="cluster", cov_kwds={"groups": model_data["uuid"]}
    )
else:
    model = smf.ols(formula, data=model_data).fit()

beta = model.params.get("reader_view", np.nan)
se = model.bse.get("reader_view", np.nan)
reg_p = model.pvalues.get("reader_view", np.nan)

# Convert log effect to percent change
pct_change = (np.expm1(beta) * 100) if np.isfinite(beta) else np.nan

results = {
    "n_dyslexia": int(sub.shape[0]),
    "n_participants": int(sub["uuid"].nunique()) if "uuid" in sub.columns else None,
    "summary": summary.reset_index().rename(columns={"index": "reader_view"}).to_dict(orient="records"),
    "mean_diff": float(mean_diff),
    "median_diff": float(median_diff),
    "welch_t": float(t_stat),
    "welch_p": float(p_val),
    "mannwhitney_u": float(u_stat) if np.isfinite(u_stat) else None,
    "mannwhitney_p": float(u_p) if np.isfinite(u_p) else None,
    "reg_beta_log": float(beta) if np.isfinite(beta) else None,
    "reg_se": float(se) if np.isfinite(se) else None,
    "reg_p": float(reg_p) if np.isfinite(reg_p) else None,
    "reg_pct_change": float(pct_change) if np.isfinite(pct_change) else None,
}

print(json.dumps(results, indent=2))
