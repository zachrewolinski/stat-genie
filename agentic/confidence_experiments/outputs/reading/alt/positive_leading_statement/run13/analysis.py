import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Filter to dyslexia participants
# Use dyslexia_bin when available; fallback to dyslexia>0
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
else:
    dys_df = df[df["dyslexia"] > 0].copy()

# Keep required columns
cols_needed = ["uuid", "reader_view", "speed", "page_id"]
dys_df = dys_df[[c for c in cols_needed if c in dys_df.columns]].copy()

# Drop missing
dys_df = dys_df.dropna(subset=["reader_view", "speed", "uuid"])

# Basic counts
n_rows = len(dys_df)
unique_participants = dys_df["uuid"].nunique()

# Summary by reader_view
summary = dys_df.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

# Per-uuid per condition mean speeds
per_uuid = (
    dys_df.groupby(["uuid", "reader_view"])["speed"]
    .mean()
    .reset_index()
)

# Identify participants with both conditions
pivot = per_uuid.pivot(index="uuid", columns="reader_view", values="speed")
paired = pivot.dropna()

paired_n = len(paired)

# Paired t-test (if enough participants)
paired_result = None
if paired_n >= 3:
    t_stat, p_val = stats.ttest_rel(paired[1], paired[0])
    diff = (paired[1] - paired[0]).mean()
    sd_diff = (paired[1] - paired[0]).std(ddof=1)
    dz = diff / sd_diff if sd_diff and not np.isnan(sd_diff) else np.nan
    paired_result = {
        "n": paired_n,
        "mean_diff": diff,
        "t_stat": t_stat,
        "p_val": p_val,
        "effect_size_dz": dz,
    }

# Independent t-test on per-uuid means (unpaired)
# Use per-uuid averages to reduce within-person dependence
per_uuid_means = per_uuid.copy()
rv0 = per_uuid_means[per_uuid_means["reader_view"] == 0]["speed"]
rv1 = per_uuid_means[per_uuid_means["reader_view"] == 1]["speed"]
ind_result = None
if len(rv0) >= 3 and len(rv1) >= 3:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    diff = rv1.mean() - rv0.mean()
    # Cohen's d (pooled SD)
    n1, n0 = len(rv1), len(rv0)
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2)) if (n1+n0-2) > 0 else np.nan
    d = diff / pooled if pooled and not np.isnan(pooled) else np.nan
    ind_result = {
        "n1": n1,
        "n0": n0,
        "mean_diff": diff,
        "t_stat": t_stat,
        "p_val": p_val,
        "effect_size_d": d,
    }

# Cluster-robust OLS on row-level data controlling for page_id
ols_result = None
if dys_df["reader_view"].nunique() == 2:
    try:
        model = smf.ols("speed ~ reader_view + C(page_id)", data=dys_df).fit(
            cov_type="cluster", cov_kwds={"groups": dys_df["uuid"]}
        )
        coef = model.params.get("reader_view", np.nan)
        p_val = model.pvalues.get("reader_view", np.nan)
        ols_result = {
            "coef": coef,
            "p_val": p_val,
            "n_obs": int(model.nobs),
        }
    except Exception as e:
        ols_result = {"error": str(e)}

# Log-speed model for robustness
log_ols_result = None
try:
    dys_df = dys_df.copy()
    dys_df = dys_df[dys_df["speed"] > 0]
    dys_df["log_speed"] = np.log(dys_df["speed"])
    model = smf.ols("log_speed ~ reader_view + C(page_id)", data=dys_df).fit(
        cov_type="cluster", cov_kwds={"groups": dys_df["uuid"]}
    )
    coef = model.params.get("reader_view", np.nan)
    p_val = model.pvalues.get("reader_view", np.nan)
    log_ols_result = {
        "coef": coef,
        "p_val": p_val,
        "n_obs": int(model.nobs),
        "pct_change": (np.exp(coef) - 1.0) if not np.isnan(coef) else np.nan,
    }
except Exception as e:
    log_ols_result = {"error": str(e)}

# Save results for inspection
results = {
    "n_rows": n_rows,
    "unique_participants": unique_participants,
    "summary_by_reader_view": summary.to_dict(orient="records"),
    "paired_test": paired_result,
    "ind_test": ind_result,
    "ols_cluster": ols_result,
    "log_ols_cluster": log_ols_result,
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
