import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = "reading.csv"
df = pd.read_csv(csv_path)

# Filter dyslexia participants
# Use dyslexia_bin if available; fallback to dyslexia >= 1
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
else:
    dys_df = df[df["dyslexia"] >= 1].copy()

# Basic counts
n_rows = len(dys_df)
unique_participants = dys_df["uuid"].nunique() if "uuid" in dys_df.columns else np.nan

# Ensure reader_view is binary
# Some entries might be floats; normalize to 0/1
rv = dys_df["reader_view"].round().astype(int)
dys_df = dys_df.assign(reader_view_bin=rv)

# Prepare speed
# Avoid non-positive speeds for log transform
speed = dys_df["speed"].astype(float)
valid_speed = speed > 0

dys_df = dys_df[valid_speed].copy()
dys_df["log_speed"] = np.log(dys_df["speed"].astype(float))

# Participant-level averages by condition
avg_by_participant = (
    dys_df.groupby(["uuid", "reader_view_bin"], as_index=False)["speed"].mean()
)

# Wide format for paired comparison
wide = avg_by_participant.pivot(index="uuid", columns="reader_view_bin", values="speed")
paired = wide.dropna()

paired_n = len(paired)

# Paired t-test on log speed using per-participant averages
# Compute log of averages (mean of logs vs log of means). We'll use log of means for interpretability.
paired_log = np.log(paired)

if paired_n > 1:
    t_stat, p_val = stats.ttest_rel(paired_log[1], paired_log[0])
    diff = paired_log[1] - paired_log[0]
    cohen_dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
else:
    t_stat, p_val, cohen_dz = np.nan, np.nan, np.nan

# Effect in percent change (geometric mean ratio)
# exp(mean log diff) - 1
pct_change = (np.exp((paired_log[1] - paired_log[0]).mean()) - 1) if paired_n > 0 else np.nan

# OLS with page fixed effects and cluster-robust SE by participant
# Use log_speed to stabilize variance
ols_result = None
ols_summary = None
ols_pval = None
ols_coef = None

try:
    # Some columns may be missing; use page_id if present else num_words
    formula = "log_speed ~ reader_view_bin"
    if "page_id" in dys_df.columns:
        formula += " + C(page_id)"
    elif "num_words" in dys_df.columns:
        formula += " + num_words"
    
    model = smf.ols(formula, data=dys_df)
    # Cluster-robust SE by uuid
    res = model.fit(cov_type="cluster", cov_kwds={"groups": dys_df["uuid"]})
    ols_result = res
    ols_pval = res.pvalues.get("reader_view_bin", np.nan)
    ols_coef = res.params.get("reader_view_bin", np.nan)
    ols_summary = res.summary().as_text()
except Exception:
    ols_summary = None

# Descriptive stats by reader_view
desc = dys_df.groupby("reader_view_bin")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Save analysis outputs to a json for later reading (optional)
analysis_out = {
    "dyslexia_rows": int(n_rows),
    "dyslexia_participants": int(unique_participants),
    "paired_participants": int(paired_n),
    "paired_t_stat": float(t_stat) if np.isfinite(t_stat) else None,
    "paired_p_value": float(p_val) if np.isfinite(p_val) else None,
    "paired_cohen_dz": float(cohen_dz) if np.isfinite(cohen_dz) else None,
    "paired_pct_change": float(pct_change) if np.isfinite(pct_change) else None,
    "ols_reader_view_coef": float(ols_coef) if np.isfinite(ols_coef) else None,
    "ols_reader_view_p_value": float(ols_pval) if np.isfinite(ols_pval) else None,
    "desc": desc.to_dict(orient="records"),
}

print(json.dumps(analysis_out, indent=2))
