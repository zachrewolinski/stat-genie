import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'

df = pd.read_csv(path)

# Define reading speed (words per minute) using reading time without scrolling
# feature5: reading time without scrolling in milliseconds
# feature7: number of words on the page
speed_wpm = df['feature7'] / (df['feature5'] / 60000.0)

df = df.copy()

df['speed_wpm'] = speed_wpm

# Filter dyslexic participants: feature17 = 1 indicates dyslexia
# Exclude non-positive or missing speeds
subset = df[(df['feature17'] == 1) & df['speed_wpm'].replace([np.inf, -np.inf], np.nan).notna()]
subset = subset[subset['speed_wpm'] > 0]

# Determine participant-level averages per condition
# feature3: reader view (1) vs not (0)

# pivot to get per participant mean speed per condition
pivot = subset.pivot_table(index='feature1', columns='feature3', values='speed_wpm', aggfunc='mean')

# participants with both conditions
paired = pivot.dropna()

# compute paired differences (reader view on - off)
diff = paired[1] - paired[0]

# Paired t-test
if len(diff) > 1:
    t_stat, p_value = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
else:
    t_stat, p_value = np.nan, np.nan

# Wilcoxon signed-rank test (if enough pairs)
try:
    if len(diff) > 0:
        w_stat, w_p = stats.wilcoxon(diff)
    else:
        w_stat, w_p = np.nan, np.nan
except ValueError:
    w_stat, w_p = np.nan, np.nan

# Effect size: Cohen's d for paired samples (mean diff / std diff)
mean_diff = diff.mean()
std_diff = diff.std(ddof=1)
cohens_d = mean_diff / std_diff if std_diff and not np.isnan(std_diff) else np.nan

# Descriptive stats
mean_on = paired[1].mean()
mean_off = paired[0].mean()
median_on = paired[1].median()
median_off = paired[0].median()

# Also compute overall (not paired) mean differences
mean_on_all = subset[subset['feature3'] == 1]['speed_wpm'].mean()
mean_off_all = subset[subset['feature3'] == 0]['speed_wpm'].mean()

# Print summary
print('Dyslexic subset rows:', len(subset))
print('Unique dyslexic participants:', subset['feature1'].nunique())
print('Participants with both conditions:', len(paired))
print('Mean speed (paired): reader view on', mean_on, 'off', mean_off)
print('Median speed (paired): on', median_on, 'off', median_off)
print('Mean diff (on - off):', mean_diff)
print('Cohen d (paired):', cohens_d)
print('Paired t-test: t', t_stat, 'p', p_value)
print('Wilcoxon: W', w_stat, 'p', w_p)
print('Mean speed (all trials): on', mean_on_all, 'off', mean_off_all)

# Bootstrap CI for mean diff
rng = np.random.default_rng(123)
if len(diff) > 5:
    boot = []
    for _ in range(5000):
        sample = rng.choice(diff.values, size=len(diff), replace=True)
        boot.append(sample.mean())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
else:
    ci_low, ci_high = np.nan, np.nan
print('Bootstrap 95% CI for mean diff:', ci_low, ci_high)

# Save key outputs to a small file for reference
out = {
    'n_rows': int(len(subset)),
    'n_participants': int(subset['feature1'].nunique()),
    'n_paired': int(len(paired)),
    'mean_on_paired': float(mean_on),
    'mean_off_paired': float(mean_off),
    'median_on_paired': float(median_on),
    'median_off_paired': float(median_off),
    'mean_diff': float(mean_diff),
    'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else None,
    't_stat': float(t_stat) if not np.isnan(t_stat) else None,
    'p_value': float(p_value) if not np.isnan(p_value) else None,
    'wilcoxon_p': float(w_p) if not np.isnan(w_p) else None,
    'mean_on_all': float(mean_on_all),
    'mean_off_all': float(mean_off_all),
    'ci_low': float(ci_low) if not np.isnan(ci_low) else None,
    'ci_high': float(ci_high) if not np.isnan(ci_high) else None,
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(out, f, indent=2)

