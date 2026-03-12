import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Filter dyslexia individuals
# Prefer dyslexia_bin if present; otherwise use dyslexia >= 1
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    dys_df = df[df['dyslexia'] >= 1].copy()

# Basic counts
counts = dys_df['reader_view'].value_counts(dropna=False)

# Clean speed values
# Remove non-positive or missing speeds
speeds = dys_df[['reader_view', 'speed', 'uuid']].dropna()
speeds = speeds[speeds['speed'] > 0]

# Split groups
speed_rv1 = speeds.loc[speeds['reader_view'] == 1, 'speed']
speed_rv0 = speeds.loc[speeds['reader_view'] == 0, 'speed']

# Welch t-test
welch = stats.ttest_ind(speed_rv1, speed_rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
if len(speed_rv1) > 0 and len(speed_rv0) > 0:
    mwu = stats.mannwhitneyu(speed_rv1, speed_rv0, alternative='two-sided')
else:
    mwu = None

# Effect size (Cohen's d for unequal n)
# pooled SD using unbiased variance
n1, n0 = len(speed_rv1), len(speed_rv0)
mean1, mean0 = speed_rv1.mean(), speed_rv0.mean()
var1, var0 = speed_rv1.var(ddof=1), speed_rv0.var(ddof=1)
# if variance undefined (n<2) guard
if n1 > 1 and n0 > 1:
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
    d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
else:
    d = np.nan

# Within-subject paired analysis for participants with both conditions
# Compute per-uuid mean speed for each reader_view
pivot = speeds.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna()  # has both 0 and 1
paired_n = len(paired)
if paired_n > 1:
    paired_diff = paired[1] - paired[0]
    paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    # Effect size for paired samples (Cohen's dz)
    dz = paired_diff.mean() / paired_diff.std(ddof=1) if paired_diff.std(ddof=1) > 0 else np.nan
else:
    paired_t = None
    dz = np.nan
    paired_diff = pd.Series(dtype=float)

# Also compute median/mean speeds for context
summary = {
    'n_total_dyslexia': int(len(dys_df)),
    'n_speed_positive': int(len(speeds)),
    'n_reader_view_1': int(n1),
    'n_reader_view_0': int(n0),
    'mean_speed_rv1': float(mean1) if n1 > 0 else np.nan,
    'mean_speed_rv0': float(mean0) if n0 > 0 else np.nan,
    'median_speed_rv1': float(speed_rv1.median()) if n1 > 0 else np.nan,
    'median_speed_rv0': float(speed_rv0.median()) if n0 > 0 else np.nan,
    'welch_t': float(welch.statistic) if n1 > 1 and n0 > 1 else np.nan,
    'welch_p': float(welch.pvalue) if n1 > 1 and n0 > 1 else np.nan,
    'mwu_u': float(mwu.statistic) if mwu else np.nan,
    'mwu_p': float(mwu.pvalue) if mwu else np.nan,
    'cohens_d': float(d),
    'paired_n': int(paired_n),
    'paired_mean_diff': float(paired_diff.mean()) if paired_n > 0 else np.nan,
    'paired_median_diff': float(paired_diff.median()) if paired_n > 0 else np.nan,
    'paired_t': float(paired_t.statistic) if paired_t else np.nan,
    'paired_p': float(paired_t.pvalue) if paired_t else np.nan,
    'paired_dz': float(dz),
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
