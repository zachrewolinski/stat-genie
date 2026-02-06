import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('reading.csv')

# Identify dyslexic participants (feature17 == 1)
dys = df[df['feature17'] == 1].copy()

# Reading speed is feature20 (words per minute), derived from words / reading time
speed_col = 'feature20'

# Compute per-participant mean speed by condition (reader view on/off)
pp = dys.groupby(['feature1', 'feature3'])[speed_col].mean().unstack()
pp = pp.dropna()  # keep participants with both conditions

# Paired t-test on participant-level means
res = stats.ttest_rel(pp[1], pp[0], nan_policy='omit')

diff = pp[1] - pp[0]
cohen_d = diff.mean() / diff.std(ddof=1)

summary = {
    'n_participants': int(len(pp)),
    'mean_speed_no_reader_view': float(dys[dys['feature3'] == 0][speed_col].mean()),
    'mean_speed_reader_view': float(dys[dys['feature3'] == 1][speed_col].mean()),
    'mean_diff_reader_minus_no': float(diff.mean()),
    'median_diff_reader_minus_no': float(diff.median()),
    't_stat': float(res.statistic),
    'p_value': float(res.pvalue),
    'cohen_d_paired': float(cohen_d),
}

print(summary)
