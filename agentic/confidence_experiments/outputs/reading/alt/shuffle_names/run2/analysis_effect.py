import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Define dyslexia group: dyslexia values 1 or 2

df_dys = df[df['dyslexia'].isin([1.0, 2.0])].copy()

# Keep necessary columns

df_dys = df_dys[['speed','language','running_time','num_words','adjusted_running_time','age']]
# drop missing

df_dys = df_dys.dropna(subset=['language','running_time'])

# Reader view indicator

df_dys['reader_view'] = df_dys['language']

# Group stats

group_stats = df_dys.groupby('reader_view')['running_time'].describe()
print('group stats running_time')
print(group_stats)

# Independent t-test (Welch)

g0 = df_dys[df_dys['reader_view']==0]['running_time']
g1 = df_dys[df_dys['reader_view']==1]['running_time']
t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')

# Effect size (Cohen d, using pooled SD)

def cohens_d(a,b):
    a = a.dropna()
    b = b.dropna()
    n1 = len(a)
    n2 = len(b)
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / s_pooled


d = cohens_d(g1, g0)
print('Welch t-test:', t_stat, p_val, 'Cohen d', d)

# Non-parametric test (Mann-Whitney U)

u_stat, p_u = stats.mannwhitneyu(g1, g0, alternative='two-sided')
print('Mann-Whitney U:', u_stat, p_u)

# Paired analysis: participants with both conditions

pivot = df_dys.pivot_table(index='speed', columns='reader_view', values='running_time', aggfunc='mean')
paired = pivot.dropna(subset=[0,1])
print('paired participants', paired.shape)
if len(paired) > 1:
    t_stat_p, p_val_p = stats.ttest_rel(paired[1], paired[0])
    diff = paired[1] - paired[0]
    print('paired t-test', t_stat_p, p_val_p, 'mean diff', diff.mean())

# sensitivity: compute reading speed from adjusted_running_time assuming centiseconds

df_dys['wpm_centi'] = df_dys['num_words'] * 6000 / df_dys['adjusted_running_time']
g0_w = df_dys[df_dys['reader_view']==0]['wpm_centi']
g1_w = df_dys[df_dys['reader_view']==1]['wpm_centi']
t_stat_w, p_val_w = stats.ttest_ind(g1_w, g0_w, equal_var=False, nan_policy='omit')
d_w = cohens_d(g1_w, g0_w)
print('wpm_centi Welch t-test', t_stat_w, p_val_w, 'd', d_w)
