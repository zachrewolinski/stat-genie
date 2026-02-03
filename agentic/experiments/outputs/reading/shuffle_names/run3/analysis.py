import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind

# Load data
DATA_PATH = 'reading.csv'

df = pd.read_csv(DATA_PATH)

# Map shuffled columns to their semantic meaning (inferred from metadata + data relationships)
# reader_view indicator (0/1) appears in `language`
df['reader_view_flag'] = df['language']

# Dyslexia status is encoded in `device` (0 = no dyslexia, 1/2 = dyslexia/severe)
df['dyslexic'] = df['device'] > 0

# Reading speed is stored in `running_time` (matches words/min computed from num_words/adjusted time)
df['reading_speed'] = df['running_time']

# Basic sanity checks
assert set(df['reader_view_flag'].dropna().unique()).issubset({0, 1})

# Filter to individuals with dyslexia
dys = df[df['dyslexic']].copy()

# Group stats
stats = dys.groupby('reader_view_flag')['reading_speed'].agg(['count', 'mean', 'median', 'std'])

# Welch t-test on reading speed between reader view on/off
rv_on = dys.loc[dys['reader_view_flag'] == 1, 'reading_speed'].dropna()
rv_off = dys.loc[dys['reader_view_flag'] == 0, 'reading_speed'].dropna()

# If either group is empty, skip test
if len(rv_on) > 1 and len(rv_off) > 1:
    t_stat, p_value, dfree = ttest_ind(rv_on, rv_off, usevar='unequal')
else:
    t_stat, p_value, dfree = np.nan, np.nan, np.nan

# Effect size (difference in means and percent change vs control)
mean_on = rv_on.mean()
mean_off = rv_off.mean()
mean_diff = mean_on - mean_off
pct_change = mean_diff / mean_off * 100 if mean_off != 0 else np.nan

print('Dyslexic subgroup reading speed by reader view:')
print(stats)
print('\nWelch t-test: t=%.3f, p=%.4g, dof=%.1f' % (t_stat, p_value, dfree))
print('Mean on: %.3f, mean off: %.3f, diff: %.3f (%.2f%%)' % (mean_on, mean_off, mean_diff, pct_change))
