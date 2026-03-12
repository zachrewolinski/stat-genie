import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# assumed mapping based on value patterns
reader_view = 'language'          # binary 0/1
speed = 'running_time'            # continuous speed-like measure

# dyslexia status (0=no,1=dyslexia,2=severe)
mask_dys = df['device'].isin([1.0,2.0])
sub = df.loc[mask_dys, [reader_view, speed]].dropna()

rv0 = sub.loc[sub[reader_view]==0, speed]
rv1 = sub.loc[sub[reader_view]==1, speed]

n0, n1 = len(rv0), len(rv1)
mean0, mean1 = rv0.mean(), rv1.mean()
med0, med1 = rv0.median(), rv1.median()
std0, std1 = rv0.std(ddof=1), rv1.std(ddof=1)

# Welch t-test
if n0>1 and n1>1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
else:
    t_stat, p_val = np.nan, np.nan

# Cohen's d (pooled)
pooled_sd = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2)/(n1+n0-2)) if (n0+n1>2) else np.nan
cohen_d = (mean1-mean0)/pooled_sd if pooled_sd and pooled_sd>0 else np.nan

# Mann-Whitney
u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')

print('n0', n0, 'n1', n1)
print('mean0', mean0, 'mean1', mean1)
print('median0', med0, 'median1', med1)
print('std0', std0, 'std1', std1)
print('t', t_stat, 'p', p_val)
print('cohen_d', cohen_d)
print('mw_p', p_u)
