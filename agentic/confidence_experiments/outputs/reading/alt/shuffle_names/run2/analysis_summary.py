import pandas as pd
import numpy as np
from scipy import stats

# Load

df = pd.read_csv('reading.csv')

# Dyslexia group (1 or 2)

df_dys = df[df['dyslexia'].isin([1.0, 2.0])].copy()

# reader view indicator is 'language' (0/1)
# drop missing

df_dys = df_dys.dropna(subset=['language', 'running_time', 'num_words', 'adjusted_running_time'])

# define groups

g0 = df_dys[df_dys['language']==0]
g1 = df_dys[df_dys['language']==1]

# Outcome 1: running_time (speed proxy)

mean0 = g0['running_time'].mean()
mean1 = g1['running_time'].mean()
med0 = g0['running_time'].median()
med1 = g1['running_time'].median()

# Welch t-test

t_stat, p_val = stats.ttest_ind(g1['running_time'], g0['running_time'], equal_var=False, nan_policy='omit')

# effect size

n1 = g1['running_time'].dropna()
n0 = g0['running_time'].dropna()

def cohens_d(a,b):
    a = a.dropna()
    b = b.dropna()
    n1 = len(a)
    n2 = len(b)
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / s_pooled


d = cohens_d(g1['running_time'], g0['running_time'])

# paired
pivot = df_dys.pivot_table(index='speed', columns='language', values='running_time', aggfunc='mean')
paired = pivot.dropna(subset=[0,1])
paired_t = stats.ttest_rel(paired[1], paired[0]) if len(paired) > 1 else None
paired_mean_diff = (paired[1]-paired[0]).mean() if len(paired) > 0 else None

# Outcome 2: speed from adjusted_running_time (words per time unit scaled to wpm if time is centiseconds)

wpm_centi = df_dys['num_words'] * 6000 / df_dys['adjusted_running_time']

wpm0 = wpm_centi[df_dys['language']==0]
wpm1 = wpm_centi[df_dys['language']==1]

wpm_mean0 = wpm0.mean()
wpm_mean1 = wpm1.mean()

wpm_t = stats.ttest_ind(wpm1, wpm0, equal_var=False, nan_policy='omit')

print('N dyslexia', len(df_dys), 'group0', len(g0), 'group1', len(g1))
print('running_time mean0', mean0, 'mean1', mean1, 'median0', med0, 'median1', med1)
print('running_time Welch p', p_val, 't', t_stat, 'd', d)
print('paired n', len(paired), 'paired p', (paired_t.pvalue if paired_t else None), 'mean diff', paired_mean_diff)
print('wpm mean0', wpm_mean0, 'mean1', wpm_mean1, 'p', wpm_t.pvalue)
