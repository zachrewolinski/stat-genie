import pandas as pd
import numpy as np
from scipy import stats

# Load
df = pd.read_csv('reading.csv')

# Reading speed variable
speed = df['feature20']  # words per minute (derived from words/reading time)

# Dyslexia indicator
# feature17: 1 dyslexia, 0 no dyslexia
# feature12: 0 none, 1 dyslexia, 2 severe dyslexia

print('Total rows', len(df))
print('feature17 counts', df['feature17'].value_counts(dropna=False))
print('feature12 counts', df['feature12'].value_counts(dropna=False))

# Choose dyslexia filter

dyslexic = df['feature17'] == 1

# Reader view
rv_on = df['feature3'] == 1

# Basic stats
sub = df[dyslexic].copy()
print('Dyslexic rows', len(sub))
print('Reader view counts in dyslexic', sub['feature3'].value_counts())

# Remove extreme outliers maybe? We'll compute analysis with all data and with trimmed speeds (1st-99th percentile)

# function to compute paired analysis per participant

def paired_analysis(data, speed_col='feature20'):
    # participant id feature1
    # compute per participant mean by reader view
    means = data.pivot_table(index='feature1', columns='feature3', values=speed_col, aggfunc='mean')
    # columns: 0 and 1
    both = means.dropna()
    if both.empty:
        return None
    diff = both[1] - both[0]
    # paired t-test
    tstat, pval = stats.ttest_rel(both[1], both[0])
    # effect size: Cohen's d for paired (mean diff / sd diff)
    d = diff.mean() / diff.std(ddof=1)
    return {
        'n_pairs': len(both),
        'mean_on': both[1].mean(),
        'mean_off': both[0].mean(),
        'mean_diff': diff.mean(),
        'tstat': tstat,
        'pval': pval,
        'd': d,
    }

# function for independent groups (using all rows)

def independent_analysis(data, speed_col='feature20'):
    g_on = data.loc[data['feature3'] == 1, speed_col]
    g_off = data.loc[data['feature3'] == 0, speed_col]
    tstat, pval = stats.ttest_ind(g_on, g_off, equal_var=False)
    # effect size (Hedges g)
    n1, n2 = len(g_on), len(g_off)
    mean1, mean2 = g_on.mean(), g_off.mean()
    s1, s2 = g_on.std(ddof=1), g_off.std(ddof=1)
    # pooled SD for Hedges g (unbiased)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / sp if sp > 0 else np.nan
    # Hedges correction
    J = 1 - (3 / (4*(n1 + n2) - 9))
    g = d * J
    return {
        'n_on': n1,
        'n_off': n2,
        'mean_on': mean1,
        'mean_off': mean2,
        'mean_diff': mean1 - mean2,
        'tstat': tstat,
        'pval': pval,
        'g': g,
    }

# Analysis with all data
paired_all = paired_analysis(sub)
ind_all = independent_analysis(sub)
print('paired_all', paired_all)
print('ind_all', ind_all)

# Trim outliers based on 1st-99th percentile within dyslexic group
lo, hi = sub['feature20'].quantile([0.01, 0.99])
sub_trim = sub[(sub['feature20'] >= lo) & (sub['feature20'] <= hi)]
paired_trim = paired_analysis(sub_trim)
ind_trim = independent_analysis(sub_trim)
print('trim range', lo, hi)
print('paired_trim', paired_trim)
print('ind_trim', ind_trim)

# Also using feature12>0 for dyslexia
sub2 = df[df['feature12'] > 0].copy()
paired_all2 = paired_analysis(sub2)
ind_all2 = independent_analysis(sub2)
print('feature12>0 dyslexic rows', len(sub2))
print('paired_all2', paired_all2)
print('ind_all2', ind_all2)

# Optionally compute nonparametric test (Wilcoxon) for paired
if paired_all:
    means = sub.pivot_table(index='feature1', columns='feature3', values='feature20', aggfunc='mean').dropna()
    if len(means) > 0:
        wstat, wp = stats.wilcoxon(means[1], means[0])
        print('wilcoxon paired', wstat, wp)

