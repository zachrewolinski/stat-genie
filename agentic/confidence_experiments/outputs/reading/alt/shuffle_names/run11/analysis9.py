import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

wpm = df['num_words'] / df['running_time'] * 60

# use dyslexia_bin as dyslexia indicator (1)
mask = (df['dyslexia_bin']==1) & df['language'].notna() & wpm.notna()
rv0 = wpm[mask & (df['language']==0)]
rv1 = wpm[mask & (df['language']==1)]

print('n rv0', rv0.shape[0], 'n rv1', rv1.shape[0])
print('mean rv0', rv0.mean(), 'mean rv1', rv1.mean())

if rv0.shape[0]>1 and rv1.shape[0]>1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    print('t-test', t_stat, p_val)
    u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('mannwhitney', u_stat, p_u)

    # Cohen's d
    n1, n0 = rv1.shape[0], rv0.shape[0]
    var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
    d = (rv1.mean() - rv0.mean()) / pooled
    print('cohen_d', d)
