import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Define variables based on metadata mapping
# reader_view indicator stored in 'language'
# dyslexia status stored in 'device' (0=no, 1=dyslexia, 2=severe)
# reading speed stored in 'running_time'

# filter dyslexic participants (device > 0)
sub = df[['language', 'device', 'running_time']].dropna()
sub = sub[sub['device'] > 0]

# split by reader_view
rv_on = sub[sub['language'] == 1]['running_time']
rv_off = sub[sub['language'] == 0]['running_time']

print('Counts: reader_view on', len(rv_on), 'off', len(rv_off))
print('Means: on', rv_on.mean(), 'off', rv_off.mean())
print('Medians: on', rv_on.median(), 'off', rv_off.median())
print('Std: on', rv_on.std(), 'off', rv_off.std())

# t-test (Welch)
if len(rv_on) > 1 and len(rv_off) > 1:
    tstat, pval = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
    print('Welch t-test p=', pval, 't=', tstat)

    # Cohen's d
    n1, n2 = len(rv_on), len(rv_off)
    s1, s2 = rv_on.var(ddof=1), rv_off.var(ddof=1)
    pooled = ((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2)
    d = (rv_on.mean() - rv_off.mean())/np.sqrt(pooled)
    print('Cohen d', d)

# Mann-Whitney U
try:
    ustat, upval = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
    print('Mann-Whitney p=', upval)
except Exception as e:
    print('Mann-Whitney error', e)

# also compare using log transform (to reduce skew)
rv_on_log = np.log1p(rv_on)
rv_off_log = np.log1p(rv_off)
if len(rv_on_log) > 1 and len(rv_off_log) > 1:
    tstat_log, pval_log = stats.ttest_ind(rv_on_log, rv_off_log, equal_var=False, nan_policy='omit')
    print('Welch t-test log p=', pval_log, 't=', tstat_log)

