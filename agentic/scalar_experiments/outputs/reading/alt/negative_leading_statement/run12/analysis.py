import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "reading.csv"

df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure numeric columns are numeric
for col in ["reader_view", "speed", "dyslexia_bin", "dyslexia"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter dyslexic participants
# Use dyslexia_bin == 1 as dyslexia indicator
sub = df[df["dyslexia_bin"] == 1].copy()

# Drop missing values for key columns
sub = sub.dropna(subset=["reader_view", "speed", "page_id", "uuid"])

# Log-transform speed to handle skew
sub["log_speed"] = np.log(sub["speed"])

# Summary stats
summary = (
    sub.groupby("reader_view")["speed"]
    .agg(["count", "mean", "median", "std"])
    .rename(index={0: "No Reader View", 1: "Reader View"})
)

# Nonparametric test on raw speed
rv0 = sub.loc[sub["reader_view"] == 0, "speed"]
rv1 = sub.loc[sub["reader_view"] == 1, "speed"]

mw_stat, mw_p = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")

# OLS with page fixed effects, cluster-robust SE by participant
# Using log_speed for normality
sub["reader_view"] = sub["reader_view"].astype(int)

model = smf.ols("log_speed ~ reader_view + C(page_id)", data=sub)
res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["uuid"]})

beta = res.params.get("reader_view", np.nan)
se = res.bse.get("reader_view", np.nan)
# 95% CI
ci_low = beta - 1.96 * se
ci_high = beta + 1.96 * se

# Convert log effect to percent change
pct_change = (np.exp(beta) - 1.0) * 100.0
pct_low = (np.exp(ci_low) - 1.0) * 100.0
pct_high = (np.exp(ci_high) - 1.0) * 100.0

# Two-sided p-value for reader_view
p_value = res.pvalues.get("reader_view", np.nan)

# Also run simple t-test on log_speed
rv0_log = sub.loc[sub["reader_view"] == 0, "log_speed"]
rv1_log = sub.loc[sub["reader_view"] == 1, "log_speed"]

t_stat, t_p = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")

output = {
    "n_total": int(len(sub)),
    "n_reader_view": int((sub["reader_view"] == 1).sum()),
    "n_no_reader_view": int((sub["reader_view"] == 0).sum()),
    "summary": summary.reset_index().to_dict(orient="records"),
    "mannwhitney_p": float(mw_p),
    "ttest_log_p": float(t_p),
    "ols_log_beta": float(beta),
    "ols_log_se": float(se),
    "ols_log_ci_low": float(ci_low),
    "ols_log_ci_high": float(ci_high),
    "ols_pct_change": float(pct_change),
    "ols_pct_ci_low": float(pct_low),
    "ols_pct_ci_high": float(pct_high),
    "ols_p_value": float(p_value),
}

print(json.dumps(output, indent=2))
