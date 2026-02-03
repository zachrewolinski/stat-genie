import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Interpret columns based on metadata + inspection
# language: 0/1 indicator (reader view off/on)
# dyslexia_bin: 0/1 indicator (no dyslexia/dyslexia)
# running_time: reading time in seconds (distribution aligns with typical reading times)
# num_words: number of words on the page

# Compute reading speed in words per minute
# Avoid divide-by-zero and missing values
valid = df[['language', 'dyslexia_bin', 'running_time', 'num_words']].copy()
valid = valid.dropna()
valid = valid[valid['running_time'] > 0]
valid['wpm'] = valid['num_words'] / (valid['running_time'] / 60.0)

# Focus on participants with dyslexia
sub = valid[valid['dyslexia_bin'] == 1]

# Group statistics by reader view
stats = sub.groupby('language')['wpm'].agg(['count', 'mean', 'std']).rename(index={0: 'reader_view_off', 1: 'reader_view_on'})

# Two-sample t-test (Welch) comparing reader view on vs off
rv_on = sub[sub['language'] == 1]['wpm']
rv_off = sub[sub['language'] == 0]['wpm']

# If either group is empty, avoid crashing
if len(rv_on) > 1 and len(rv_off) > 1:
    t_stat, p_value, dfree = ttest_ind(rv_on, rv_off, usevar='unequal')
else:
    t_stat, p_value, dfree = (np.nan, np.nan, np.nan)

# Print results for inspection
print('Dyslexia subset size:', len(sub))
print(stats)
print('Mean difference (on - off):', rv_on.mean() - rv_off.mean())
print('Welch t-test: t=%.4f, p=%.4f, df=%.2f' % (t_stat, p_value, dfree))

# Save a compact summary for downstream use if needed
summary = {
    'n_dyslexia': int(len(sub)),
    'mean_wpm_reader_view_on': float(rv_on.mean()),
    'mean_wpm_reader_view_off': float(rv_off.mean()),
    'mean_diff_on_minus_off': float(rv_on.mean() - rv_off.mean()),
    't_stat': float(t_stat),
    'p_value': float(p_value),
    'df': float(dfree),
}

pd.Series(summary).to_csv('analysis_summary.csv')
