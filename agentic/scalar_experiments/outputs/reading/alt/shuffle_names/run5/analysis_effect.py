import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Map columns based on inspection
# running_time behaves like reading speed (high correlation with words / time)
# language is balanced binary -> reader_view indicator
# device (0/1/2) aligns perfectly with correct_rate binary -> dyslexia status

df = _df.copy()

# Basic flags
# Dyslexic if device > 0 (1 or 2)
df = df[df['device'].notna() & df['language'].notna() & df['running_time'].notna()]

df['dyslexic'] = df['device'] > 0

dys = df[df['dyslexic']].copy()

print('Total rows', len(df), 'Dyslexic rows', len(dys))
print('Unique participants (overall)', df['speed'].nunique())
print('Unique participants (dyslexic)', dys['speed'].nunique())

# Group stats for reading speed by reader_view
summary = dys.groupby('language')['running_time'].agg(['count','mean','median','std'])
print('\nSpeed by reader_view (language) in dyslexic subset:')
print(summary)

# Welch t-test
rv0 = dys.loc[dys['language'] == 0, 'running_time']
rv1 = dys.loc[dys['language'] == 1, 'running_time']

# remove extreme outliers? We'll keep but also report trimmed results

t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('\nWelch t-test (rv1 vs rv0): t=%.3f p=%.6f' % (t_stat, p_val))

# Mann-Whitney U
u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
print('Mann-Whitney U: U=%.1f p=%.6f' % (u_stat, p_u))

# Effect size (Cohen's d)
mean_diff = rv1.mean() - rv0.mean()
# pooled SD (Welch) using sample sizes
n1, n0 = rv1.count(), rv0.count()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohen_d = mean_diff / pooled_sd
print('Mean diff (rv1 - rv0): %.3f, Cohen d: %.3f' % (mean_diff, cohen_d))

# Robust regression controlling for text length and readability and page ID
# Use cluster-robust SE by participant (speed column)
# Ensure variables are not missing
reg_df = dys[['running_time','language','num_words','uuid','scrolling_time','speed']].dropna().copy()

model = smf.ols('running_time ~ language + num_words + uuid + C(scrolling_time)', data=reg_df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': reg_df['speed']})
print('\nRegression results (cluster-robust):')
print(res.summary().tables[1])

# Store key metrics for later use

