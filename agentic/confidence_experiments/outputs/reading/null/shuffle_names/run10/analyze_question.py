import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

# Identify variables
reader_view = df['language']  # binary 0/1 indicator per metadata
# dyslexia status
# treat dyslexia>0 as dyslexic

dyslexic = df['dyslexia'] > 0

# reading speed: compute words per minute from running_time (assumed seconds)
wpm = df['num_words'] / df['running_time'] * 60

# remove non-positive or infinite
mask = np.isfinite(wpm) & (df['running_time']>0)

sub = df.loc[mask & dyslexic].copy()
sub['wpm']=wpm[mask & dyslexic]

# groups
rv_on = sub[sub['language']==1]['wpm']
rv_off = sub[sub['language']==0]['wpm']

print('dyslexic n', len(sub), 'rv_on', len(rv_on), 'rv_off', len(rv_off))
print('means', rv_on.mean(), rv_off.mean())
print('medians', rv_on.median(), rv_off.median())

# Welch t-test
if len(rv_on)>1 and len(rv_off)>1:
    tstat, pval = stats.ttest_ind(rv_on, rv_off, equal_var=False, nan_policy='omit')
    print('welch t', tstat, 'p', pval)

# Mann-Whitney
if len(rv_on)>1 and len(rv_off)>1:
    u, p = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
    print('mannwhitney p', p)

# effect size cohen's d
mean_diff = rv_on.mean() - rv_off.mean()
# pooled sd for cohen's d
s1, s2 = rv_on.std(ddof=1), rv_off.std(ddof=1)
pooled = np.sqrt(((len(rv_on)-1)*s1**2 + (len(rv_off)-1)*s2**2)/(len(rv_on)+len(rv_off)-2))
cohen_d = mean_diff/pooled if pooled>0 else np.nan
print('mean_diff', mean_diff, 'cohen_d', cohen_d)

# also compute difference in log(wpm) to reduce skew
sub['log_wpm'] = np.log(sub['wpm'])
rv_on_l = sub[sub['language']==1]['log_wpm']
rv_off_l = sub[sub['language']==0]['log_wpm']
if len(rv_on_l)>1 and len(rv_off_l)>1:
    tstat_l, pval_l = stats.ttest_ind(rv_on_l, rv_off_l, equal_var=False, nan_policy='omit')
    print('log welch t', tstat_l, 'p', pval_l, 'mean diff log', rv_on_l.mean()-rv_off_l.mean())

# summary stats for full sample as sanity
print('overall wpm stats', wpm.describe())
