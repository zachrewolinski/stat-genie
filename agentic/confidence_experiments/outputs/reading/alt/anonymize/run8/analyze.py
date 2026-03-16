import json
import pandas as pd
import numpy as np
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Column mapping from info.json
# feature3: reader view activated (1) or not (0)
# feature5: time on page minus scrolling duration (ms)
# feature7: number of words on the page
# feature12: dyslexia status (0 none, 1 dyslexia, 2 severe)
# feature17: dyslexia indicator (1 yes, 0 no)

# Prefer binary dyslexia indicator if present and non-null
if "feature17" in df.columns and df["feature17"].notna().any():
    dyslexia_mask = df["feature17"] == 1
else:
    dyslexia_mask = df["feature12"] >= 1

# Compute reading speed in words per minute using reading time minus scrolling
# Avoid division by zero or negative times
reading_time_ms = df["feature5"].astype(float)
words = df["feature7"].astype(float)
valid_time = reading_time_ms > 0
speed_wpm = (words / reading_time_ms) * 60000.0

# Clean speeds: finite and positive
speed_wpm = speed_wpm.replace([np.inf, -np.inf], np.nan)

# Add to df
work = df.copy()
work["speed_wpm"] = speed_wpm

# Filter to dyslexia group and valid speeds
sub = work[dyslexia_mask & valid_time & work["speed_wpm"].notna()]

# Reader view indicator
rv = sub["feature3"].astype(int)

# Groups
speed_rv = sub.loc[rv == 1, "speed_wpm"]
speed_no = sub.loc[rv == 0, "speed_wpm"]

# Basic stats
stats_out = {
    "n_total_dyslexia": int(sub.shape[0]),
    "n_reader_view": int(speed_rv.shape[0]),
    "n_no_reader_view": int(speed_no.shape[0]),
    "mean_rv": float(speed_rv.mean()),
    "mean_no": float(speed_no.mean()),
    "median_rv": float(speed_rv.median()),
    "median_no": float(speed_no.median()),
    "std_rv": float(speed_rv.std(ddof=1)),
    "std_no": float(speed_no.std(ddof=1)),
}

# Welch t-test
if speed_rv.shape[0] > 1 and speed_no.shape[0] > 1:
    t_stat, p_val = stats.ttest_ind(speed_rv, speed_no, equal_var=False, nan_policy="omit")
else:
    t_stat, p_val = np.nan, np.nan

# Effect size (Cohen's d) using pooled SD (unequal n)
if speed_rv.shape[0] > 1 and speed_no.shape[0] > 1:
    n1 = speed_rv.shape[0]
    n2 = speed_no.shape[0]
    s1 = speed_rv.var(ddof=1)
    s2 = speed_no.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (speed_rv.mean() - speed_no.mean()) / pooled_sd if pooled_sd > 0 else np.nan
else:
    d = np.nan

# Non-parametric test (Mann-Whitney) as robustness
if speed_rv.shape[0] > 0 and speed_no.shape[0] > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(speed_rv, speed_no, alternative="two-sided")
    except ValueError:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

results = {
    "stats": stats_out,
    "t_test": {"t_stat": float(t_stat) if np.isfinite(t_stat) else None, "p_value": float(p_val) if np.isfinite(p_val) else None},
    "cohens_d": float(d) if np.isfinite(d) else None,
    "mann_whitney": {"u_stat": float(u_stat) if np.isfinite(u_stat) else None, "p_value": float(p_u) if np.isfinite(p_u) else None},
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
