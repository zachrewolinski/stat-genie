import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

# Load data
DF_PATH = "reading.csv"
df = pd.read_csv(DF_PATH)

# Compute reading speed (words per minute) using adjusted running time (ms)
# wpm = num_words / (time_minutes)
valid = df["adjusted_running_time"].notna() & df["num_words"].notna()
df = df[valid].copy()
df["wpm"] = df["num_words"] / (df["adjusted_running_time"] / 60000.0)

# Focus on participants with dyslexia
# dyslexia_bin: 1 = dyslexia, 0 = no dyslexia
# language: 1 = reader view on, 0 = off
subset = df[(df["dyslexia_bin"] == 1) & (df["language"].isin([0, 1]))].copy()

rv_on = subset[subset["language"] == 1]["wpm"].dropna()
rv_off = subset[subset["language"] == 0]["wpm"].dropna()

# Summary statistics
summary = pd.DataFrame(
    {
        "group": ["reader_view_on", "reader_view_off"],
        "n": [rv_on.shape[0], rv_off.shape[0]],
        "mean_wpm": [rv_on.mean(), rv_off.mean()],
        "median_wpm": [rv_on.median(), rv_off.median()],
    }
)

# Welch's t-test on raw wpm
raw_t, raw_p, raw_df = ttest_ind(rv_on, rv_off, usevar="unequal")

# Welch's t-test on log wpm to reduce skew
log_on = np.log(rv_on[rv_on > 0])
log_off = np.log(rv_off[rv_off > 0])
log_t, log_p, log_df = ttest_ind(log_on, log_off, usevar="unequal")

# Cohen's d on raw wpm
n1, n0 = rv_on.shape[0], rv_off.shape[0]
var1, var0 = rv_on.var(ddof=1), rv_off.var(ddof=1)
sp = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
cohen_d = (rv_on.mean() - rv_off.mean()) / np.sqrt(sp)

print("Reading speed (wpm) for dyslexic participants")
print(summary.to_string(index=False))
print()
print(f"Welch t-test on wpm: t={raw_t:.3f}, p={raw_p:.4f}, df={raw_df:.2f}")
print(f"Welch t-test on log(wpm): t={log_t:.3f}, p={log_p:.4f}, df={log_df:.2f}")
print(f"Cohen's d (raw wpm): {cohen_d:.3f}")
