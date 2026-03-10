import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Determine dyslexia indicator
# feature17: 1 = dyslexia, 0 = no dyslexia (per metadata)
# feature12: 0 no, 1 dyslexia, 2 severe

# basic counts
print('total rows', len(df))
print('feature17 counts', df['feature17'].value_counts(dropna=False).to_dict())
print('feature12 counts', df['feature12'].value_counts(dropna=False).to_dict())

# create dyslexia binary using feature17
subset = df[df['feature17'] == 1].copy()
print('dyslexia subset rows (feature17==1):', len(subset))

# Ensure reading speed variable
speed = subset['feature20']
rv = subset['feature3']

print('reader view counts in dyslexia subset', rv.value_counts().to_dict())

# descriptive stats by reader view
stats_by_rv = subset.groupby('feature3')['feature20'].agg(['count','mean','median','std']).reset_index()
print('\nfeature20 by reader view (dyslexia)')
print(stats_by_rv)

# log transform for heavy tails
speed_pos = subset[subset['feature20'] > 0].copy()

# t-test on log speed
log_speed = np.log(speed_pos['feature20'])
rv_pos = speed_pos['feature3']
log0 = log_speed[rv_pos == 0]
log1 = log_speed[rv_pos == 1]

# t-test (Welch)
if len(log0) > 1 and len(log1) > 1:
    t_res = stats.ttest_ind(log1, log0, equal_var=False, nan_policy='omit')
    print('\nWelch t-test on log(speed):', t_res)

# Mann-Whitney U test on raw
if len(speed_pos[rv_pos==0]) > 1 and len(speed_pos[rv_pos==1]) > 1:
    u_res = stats.mannwhitneyu(speed_pos.loc[rv_pos==1,'feature20'], speed_pos.loc[rv_pos==0,'feature20'], alternative='two-sided')
    print('Mann-Whitney U on raw speed:', u_res)

# Effect sizes

def cohens_d(a, b):
    a = a.dropna()
    b = b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = a.var(ddof=1)
    s2 = b.var(ddof=1)
    s = np.sqrt(((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2))
    if s == 0:
        return np.nan
    return (a.mean() - b.mean())/s

# Cohen's d on log speed
if len(log0) > 1 and len(log1) > 1:
    d_log = cohens_d(log1, log0)
    print("Cohen's d (log speed, RV=1 minus RV=0):", d_log)

# median difference
median0 = speed_pos.loc[rv_pos==0,'feature20'].median()
median1 = speed_pos.loc[rv_pos==1,'feature20'].median()
print('Median speed RV=0', median0, 'RV=1', median1, 'diff', median1-median0)

# Sensitivity: dyslexia via feature12>=1
subset2 = df[df['feature12'].isin([1,2])].copy()
print('\nDyslexia subset via feature12>=1:', len(subset2))
print('reader view counts', subset2['feature3'].value_counts().to_dict())

stats_by_rv2 = subset2.groupby('feature3')['feature20'].agg(['count','mean','median','std']).reset_index()
print('feature20 by reader view (feature12>=1)')
print(stats_by_rv2)

speed_pos2 = subset2[subset2['feature20'] > 0].copy()
log_speed2 = np.log(speed_pos2['feature20'])
rv_pos2 = speed_pos2['feature3']
log0_2 = log_speed2[rv_pos2 == 0]
log1_2 = log_speed2[rv_pos2 == 1]

if len(log0_2) > 1 and len(log1_2) > 1:
    t_res2 = stats.ttest_ind(log1_2, log0_2, equal_var=False, nan_policy='omit')
    print('Welch t-test log(speed) feature12>=1:', t_res2)

if len(speed_pos2[rv_pos2==0]) > 1 and len(speed_pos2[rv_pos2==1]) > 1:
    u_res2 = stats.mannwhitneyu(speed_pos2.loc[rv_pos2==1,'feature20'], speed_pos2.loc[rv_pos2==0,'feature20'], alternative='two-sided')
    print('Mann-Whitney U raw speed feature12>=1:', u_res2)

if len(log0_2) > 1 and len(log1_2) > 1:
    d_log2 = cohens_d(log1_2, log0_2)
    print("Cohen's d (log speed, RV=1 minus RV=0, feature12>=1):", d_log2)

median0_2 = speed_pos2.loc[rv_pos2==0,'feature20'].median()
median1_2 = speed_pos2.loc[rv_pos2==1,'feature20'].median()
print('Median speed RV=0', median0_2, 'RV=1', median1_2, 'diff', median1_2-median0_2)

