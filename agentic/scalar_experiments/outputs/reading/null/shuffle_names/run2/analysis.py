import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Map likely variables based on value patterns
reader_view = df['language']  # binary 0/1
reading_speed = df['running_time']  # likely reading speed (wpm)

# Dyslexia flag from 3-level variable (device)
dyslexia_flag = df['device'] > 0

# Keep rows with required values
mask = reader_view.notna() & reading_speed.notna() & dyslexia_flag.notna()
sub = df.loc[mask].copy()
sub['reader_view'] = reader_view[mask]
sub['reading_speed'] = reading_speed[mask]
sub['dyslexia_flag'] = dyslexia_flag[mask]

# Focus on dyslexic individuals
sub_dys = sub[sub['dyslexia_flag']]

# Group stats
stats_table = sub_dys.groupby('reader_view')['reading_speed'].agg(['count','mean','median','std'])

# Welch t-test
rv1 = sub_dys[sub_dys['reader_view']==1]['reading_speed']
rv0 = sub_dys[sub_dys['reader_view']==0]['reading_speed']

# Guard against empty groups
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    # Mann-Whitney U
    u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    # Effect size (Hedges g)
    def hedges_g(x, y):
        nx, ny = len(x), len(y)
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        s_pooled = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
        if s_pooled == 0:
            return np.nan
        d = (x.mean() - y.mean()) / s_pooled
        # correction
        j = 1 - (3 / (4*(nx+ny)-9))
        return d * j
    g = hedges_g(rv1, rv0)
else:
    t_stat = p_val = u_stat = p_u = g = np.nan

print('dyslexia subset size', len(sub_dys))
print('reader_view counts', sub_dys['reader_view'].value_counts())
print('\nGroup stats (reading_speed)')
print(stats_table)
print('\nWelch t-test', t_stat, 'p', p_val)
print('Mann-Whitney U', u_stat, 'p', p_u)
print('Hedges g', g)

# Sensitivity: use binary dyslexia from correct_rate == 1
sub2 = df.loc[df['correct_rate'].notna() & reader_view.notna() & reading_speed.notna()].copy()
sub2['dyslexia_flag'] = sub2['correct_rate'] > 0
sub2 = sub2[sub2['dyslexia_flag']]
rv1b = sub2[sub2['language']==1]['running_time']
rv0b = sub2[sub2['language']==0]['running_time']

print('\nSensitivity using correct_rate as dyslexia_bin')
print('subset size', len(sub2))
print('counts', sub2['language'].value_counts())
if len(rv1b)>1 and len(rv0b)>1:
    t_stat2, p_val2 = stats.ttest_ind(rv1b, rv0b, equal_var=False)
    u_stat2, p_u2 = stats.mannwhitneyu(rv1b, rv0b, alternative='two-sided')
    g2 = (rv1b.mean() - rv0b.mean()) / np.sqrt(((rv1b.std(ddof=1)**2 + rv0b.std(ddof=1)**2)/2))
else:
    t_stat2 = p_val2 = u_stat2 = p_u2 = g2 = np.nan

print('means', rv1b.mean(), rv0b.mean())
print('medians', rv1b.median(), rv0b.median())
print('Welch t-test', t_stat2, 'p', p_val2)
print('Mann-Whitney U', u_stat2, 'p', p_u2)
print('Cohen d', g2)
