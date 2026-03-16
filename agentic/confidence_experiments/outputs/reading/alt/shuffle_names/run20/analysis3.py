import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')
# map columns
reader_view = df['language']  # 0/1
reading_speed = df['running_time']  # wpm-ish
# dyslexia indicator
# use correct_rate (0/1) and device (0/1/2) to check consistency
print('dyslexia indicator consistency:', (df['correct_rate'].fillna(-1) == (df['device']>0).astype(float).fillna(-1)).mean())

# create dyslexia group
mask_dys = df['correct_rate'] == 1

print('rows dys', mask_dys.sum())
print('unique participants dys', df.loc[mask_dys, 'speed'].nunique())

# condition counts
print('reader view counts in dys group:', df.loc[mask_dys, 'language'].value_counts(dropna=False))

# simple comparison
rv1 = reading_speed[mask_dys & (reader_view==1)]
rv0 = reading_speed[mask_dys & (reader_view==0)]

print('mean rv1', rv1.mean(), 'n', rv1.size)
print('mean rv0', rv0.mean(), 'n', rv0.size)

# t-test (Welch)
t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('welch t-test', t_stat, p_val)

# effect size (Cohen d) for unequal sizes
n1, n0 = rv1.size, rv0.size
s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/(n1+n0-2))
cohen_d = (rv1.mean() - rv0.mean())/sp
print('cohen d', cohen_d)

# paired within-subject if both conditions
# compute per participant mean per condition
sub = df.loc[mask_dys, ['speed','language','running_time']].dropna()
# pivot
pivot = sub.pivot_table(index='speed', columns='language', values='running_time', aggfunc='mean')
paired = pivot.dropna()
print('paired participants', paired.shape)

if paired.shape[0] > 1:
    diff = paired[1] - paired[0]
    t_stat_p, p_val_p = stats.ttest_1samp(diff, 0)
    print('paired t-test', t_stat_p, p_val_p)
    print('mean diff', diff.mean())

# nonparam test
u_stat, p_val_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
print('mannwhitney', u_stat, p_val_u)

# compute linear mixed model? skip for now

