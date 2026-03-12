import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Map variables based on metadata and value inspection
# reader_view indicator is column 'language' (0/1)
# dyslexia status is column 'device' (0=no dyslexia, 1=dyslexia, 2=severe dyslexia)
# reading speed is column 'running_time'

# Filter dyslexic participants (device in {1,2})
sub = df[df['device'].isin([1.0, 2.0]) & df['language'].notna() & df['running_time'].notna()].copy()

# Group by reader_view indicator
rv0 = sub[sub['language'] == 0]['running_time']
rv1 = sub[sub['language'] == 1]['running_time']

print('dyslexic rows', sub.shape[0])
print('reader_view=0 n', rv0.shape[0], 'mean', rv0.mean(), 'median', rv0.median(), 'std', rv0.std())
print('reader_view=1 n', rv1.shape[0], 'mean', rv1.mean(), 'median', rv1.median(), 'std', rv1.std())

# Welch t-test
if rv0.shape[0] > 1 and rv1.shape[0] > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    print('Welch t-test t', t_stat, 'p', p_val)

    # Mann-Whitney U test (nonparametric)
    try:
        u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
        print('Mann-Whitney U', u_stat, 'p', p_u)
    except ValueError as e:
        print('Mann-Whitney error', e)

    # Cohen's d (using pooled sd)
    n1, n0 = rv1.shape[0], rv0.shape[0]
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    pooled_sd = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2))
    d = (rv1.mean() - rv0.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    print('Cohen d', d)

# Robust comparison: trim outliers by winsorizing at 1st/99th percentile
lo = sub['running_time'].quantile(0.01)
hi = sub['running_time'].quantile(0.99)
sub['running_time_w'] = sub['running_time'].clip(lo, hi)
rv0_w = sub[sub['language'] == 0]['running_time_w']
rv1_w = sub[sub['language'] == 1]['running_time_w']
print('winsorized means rv0', rv0_w.mean(), 'rv1', rv1_w.mean())
if rv0_w.shape[0] > 1 and rv1_w.shape[0] > 1:
    t_stat_w, p_val_w = stats.ttest_ind(rv1_w, rv0_w, equal_var=False, nan_policy='omit')
    print('Welch t-test winsorized t', t_stat_w, 'p', p_val_w)
