import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Map columns based on value patterns
# language: reader_view indicator (0/1)
# device: dyslexia status (0=no, 1=dyslexia, 2=severe)
# running_time: reading speed

reader_view = df['language']
dyslexia_status = df['device']
reading_speed = df['running_time']

# filter dyslexic individuals
mask_dys = dyslexia_status.isin([1.0, 2.0])
sub = df.loc[mask_dys].copy()

# drop missing values for key columns
sub = sub[['language','running_time']].dropna()

print('dyslexic sample size', len(sub))
print('reader_view counts', sub['language'].value_counts(dropna=False))

# group stats
stats_table = sub.groupby('language')['running_time'].agg(['count','mean','median','std'])
print('\nGroup stats')
print(stats_table)

# t-test (Welch)
rv0 = sub.loc[sub['language']==0, 'running_time']
rv1 = sub.loc[sub['language']==1, 'running_time']

# check if both groups non-empty
if len(rv0) > 1 and len(rv1) > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    # effect size Cohen's d (using pooled std)
    n1, n0 = len(rv1), len(rv0)
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    pooled_sd = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/(n1+n0-2)) if n1+n0>2 else np.nan
    d = (rv1.mean() - rv0.mean())/pooled_sd if pooled_sd and pooled_sd>0 else np.nan
    print('\nWelch t-test: t=%.3f p=%.4g' % (t_stat, p_val))
    print('Cohen d', d)

    # Mann-Whitney U test (non-param)
    u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('Mann-Whitney U p=%.4g' % p_u)

# log transform to reduce skew
log_rv0 = np.log1p(rv0)
log_rv1 = np.log1p(rv1)
if len(log_rv0) > 1 and len(log_rv1) > 1:
    t_stat_log, p_val_log = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy='omit')
    print('Welch t-test on log1p: t=%.3f p=%.4g' % (t_stat_log, p_val_log))

