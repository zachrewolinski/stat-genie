import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
file_path = "reading.csv"
df = pd.read_csv(file_path)

# Key columns
participant_col = "feature1"
reader_view_col = "feature3"  # 1 = reader view, 0 = normal
speed_col = "feature20"  # reading speed (words per minute)
dyslexia_col = "feature17"  # 1 = dyslexia, 0 = no dyslexia

# Filter to dyslexia participants
sub = df[df[dyslexia_col] == 1].copy()
sub = sub[[participant_col, reader_view_col, speed_col]].dropna()

# Basic counts
n_rows = len(sub)
participants = sub[participant_col].nunique()

# Per-condition overall stats (rows)
cond_stats = (
    sub.groupby(reader_view_col)[speed_col]
    .agg(["count", "mean", "std", "median"])
    .rename(index={0: "no_reader_view", 1: "reader_view"})
)

# Per-participant mean speed by condition
pp = (
    sub.groupby([participant_col, reader_view_col])[speed_col]
    .mean()
    .unstack(reader_view_col)
)

# Keep participants with both conditions
pp_both = pp.dropna()

paired_n = len(pp_both)

# Paired t-test
if paired_n >= 2:
    diff = pp_both[1] - pp_both[0]
    t_stat, p_val = stats.ttest_rel(pp_both[1], pp_both[0])
    diff_mean = diff.mean()
    diff_std = diff.std(ddof=1)
    cohen_d = diff_mean / diff_std if diff_std and not np.isnan(diff_std) else np.nan
    # 95% CI for mean difference
    se = diff_std / np.sqrt(paired_n) if diff_std and not np.isnan(diff_std) else np.nan
    if se and not np.isnan(se):
        ci_low, ci_high = stats.t.interval(0.95, df=paired_n-1, loc=diff_mean, scale=se)
    else:
        ci_low, ci_high = np.nan, np.nan
else:
    t_stat = p_val = diff_mean = diff_std = cohen_d = ci_low = ci_high = np.nan

# Welch t-test on all rows (sensitivity)
rv = sub[sub[reader_view_col] == 1][speed_col]
no = sub[sub[reader_view_col] == 0][speed_col]
if len(rv) > 1 and len(no) > 1:
    t_welch, p_welch = stats.ttest_ind(rv, no, equal_var=False)
else:
    t_welch = p_welch = np.nan

summary = {
    "n_rows": int(n_rows),
    "n_participants": int(participants),
    "cond_stats": cond_stats.reset_index().to_dict(orient="records"),
    "paired_n": int(paired_n),
    "paired_t": float(t_stat) if pd.notna(t_stat) else None,
    "paired_p": float(p_val) if pd.notna(p_val) else None,
    "mean_diff_reader_minus_no": float(diff_mean) if pd.notna(diff_mean) else None,
    "cohen_d_paired": float(cohen_d) if pd.notna(cohen_d) else None,
    "ci95_diff": [float(ci_low), float(ci_high)] if pd.notna(ci_low) else None,
    "welch_t": float(t_welch) if pd.notna(t_welch) else None,
    "welch_p": float(p_welch) if pd.notna(p_welch) else None,
    "mean_speed_reader_view": float(rv.mean()) if len(rv) else None,
    "mean_speed_no_reader": float(no.mean()) if len(no) else None,
}

print(json.dumps(summary, indent=2))
