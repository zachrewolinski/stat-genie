import pandas as pd
import numpy as np
from scipy import stats

path = 'reading.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df))
print('columns', df.columns.tolist())

# Compute reading speed from feature7 (words) and feature5 (reading time minus scrolling) in ms.
# WPM = words / (time_minutes). time_minutes = ms / 60000

if {'feature7','feature5'}.issubset(df.columns):
    df['wpm_calc'] = df['feature7'] / (df['feature5'] / 60000.0)
    print('wpm_calc summary', df['wpm_calc'].describe())

# Feature20 summary
print('feature20 summary', df['feature20'].describe())

# Correlation between feature20 and wpm_calc
if 'wpm_calc' in df.columns:
    corr = df[['feature20','wpm_calc']].corr().iloc[0,1]
    print('corr feature20 vs wpm_calc', corr)

# define dyslexia subset using feature17==1

if 'feature17' in df.columns:
    dys = df[df['feature17'] == 1]
    print('dyslexic rows', len(dys))
    # compare reader view vs not among dyslexic
    g1 = dys[dys['feature3'] == 1]['feature20']
    g0 = dys[dys['feature3'] == 0]['feature20']
    print('feature20 dyslexic reader view n', len(g1), 'no view n', len(g0))
    print('means', g1.mean(), g0.mean())
    # Welch t-test
    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    print('t-test feature20', tstat, pval)

    # nonparametric test
    ustat, up = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    print('mannwhitney', ustat, up)

    # effect size (Cohen's d)
    def cohend(a,b):
        a = a.dropna()
        b = b.dropna()
        na = len(a); nb = len(b)
        sa = a.var(ddof=1); sb = b.var(ddof=1)
        s_pooled = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
        return (a.mean() - b.mean()) / s_pooled
    print('cohen d', cohend(g1,g0))

    # For wpm_calc if available
    if 'wpm_calc' in dys.columns:
        g1w = dys[dys['feature3']==1]['wpm_calc']
        g0w = dys[dys['feature3']==0]['wpm_calc']
        print('wpm_calc means', g1w.mean(), g0w.mean())
        tstat, pval = stats.ttest_ind(g1w, g0w, equal_var=False, nan_policy='omit')
        print('t-test wpm_calc', tstat, pval)
        ustat, up = stats.mannwhitneyu(g1w, g0w, alternative='two-sided')
        print('mannwhitney wpm_calc', ustat, up)
        print('cohen d wpm_calc', cohend(g1w, g0w))

# Also check feature12>0 as dyslexia status
if 'feature12' in df.columns:
    dys2 = df[df['feature12'] > 0]
    print('dyslexic (feature12>0) rows', len(dys2))
    g1 = dys2[dys2['feature3'] == 1]['feature20']
    g0 = dys2[dys2['feature3'] == 0]['feature20']
    print('feature20 dys2 reader view n', len(g1), 'no view n', len(g0))
    print('means', g1.mean(), g0.mean())
    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    print('t-test feature20 dys2', tstat, pval)

    if 'wpm_calc' in dys2.columns:
        g1w = dys2[dys2['feature3']==1]['wpm_calc']
        g0w = dys2[dys2['feature3']==0]['wpm_calc']
        print('wpm_calc means dys2', g1w.mean(), g0w.mean())
        tstat, pval = stats.ttest_ind(g1w, g0w, equal_var=False, nan_policy='omit')
        print('t-test wpm_calc dys2', tstat, pval)
