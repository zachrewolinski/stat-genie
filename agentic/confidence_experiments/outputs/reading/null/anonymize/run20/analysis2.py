import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Derived reading speed (words per minute)
# Using reading time minus scrolling (feature5)
# Avoid divide by zero
speed_wpm = df['feature7'] / (df['feature5'] / 60000.0)

# Clean
speed_wpm = speed_wpm.replace([np.inf, -np.inf], np.nan)

# Attach
DF = df.copy()
DF['speed_wpm'] = speed_wpm

# Dyslexia group
sub = DF[DF['feature17'] == 1].copy()

# Reader view on/off
on = sub[sub['feature3'] == 1]['speed_wpm'].dropna()
off = sub[sub['feature3'] == 0]['speed_wpm'].dropna()

# Basic stats
print('Dyslexia feature17==1')
print('n on', len(on), 'n off', len(off))
print('mean on', on.mean(), 'mean off', off.mean())
print('median on', on.median(), 'median off', off.median())
print('std on', on.std(ddof=1), 'std off', off.std(ddof=1))

# Welch t-test
print('Welch t-test', stats.ttest_ind(on, off, equal_var=False))
# Mann-Whitney
print('Mann-Whitney', stats.mannwhitneyu(on, off, alternative='two-sided'))

# Effect size Cohen's d (pooled) and Hedges g
n1, n2 = len(on), len(off)
mean1, mean2 = on.mean(), off.mean()
s1, s2 = on.std(ddof=1), off.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
cohen_d = (mean1-mean2)/sp
# Hedges g correction
J = 1 - 3/(4*(n1+n2)-9)
hedges_g = J * cohen_d
print('cohen_d', cohen_d, 'hedges_g', hedges_g)

# Trimmed mean (5%)
trim = 0.05
trim_mean_on = stats.trim_mean(on, proportiontocut=trim)
trim_mean_off = stats.trim_mean(off, proportiontocut=trim)
print('trimmed mean 5% on', trim_mean_on, 'off', trim_mean_off)

# Log transform (add small constant)
log_on = np.log(on + 1)
log_off = np.log(off + 1)
print('log mean on', log_on.mean(), 'off', log_off.mean())
print('log Welch t-test', stats.ttest_ind(log_on, log_off, equal_var=False))

# Bootstrap difference in means (raw) and medians
rng = np.random.default_rng(0)
boot = 10000

def bootstrap_diff(a, b, func=np.mean):
    diffs = []
    for _ in range(boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs.append(func(a_s) - func(b_s))
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])

mean_diff, mean_ci = bootstrap_diff(on.values, off.values, np.mean)
med_diff, med_ci = bootstrap_diff(on.values, off.values, np.median)
print('bootstrap mean diff', mean_diff, mean_ci)
print('bootstrap median diff', med_diff, med_ci)

# Additional check: using feature12>=1
sub2 = DF[DF['feature12'] >= 1].copy()
on2 = sub2[sub2['feature3'] == 1]['speed_wpm'].dropna()
off2 = sub2[sub2['feature3'] == 0]['speed_wpm'].dropna()
print('\nDyslexia feature12>=1')
print('n on', len(on2), 'n off', len(off2))
print('mean on', on2.mean(), 'mean off', off2.mean())
print('median on', on2.median(), 'median off', off2.median())
print('Welch t-test', stats.ttest_ind(on2, off2, equal_var=False))
print('Mann-Whitney', stats.mannwhitneyu(on2, off2, alternative='two-sided'))

# Effect size for feature12>=1
n1, n2 = len(on2), len(off2)
mean1, mean2 = on2.mean(), off2.mean()
s1, s2 = on2.std(ddof=1), off2.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
cohen_d2 = (mean1-mean2)/sp
J2 = 1 - 3/(4*(n1+n2)-9)
hedges_g2 = J2 * cohen_d2
print('cohen_d', cohen_d2, 'hedges_g', hedges_g2)

# Log transform
log_on2 = np.log(on2 + 1)
log_off2 = np.log(off2 + 1)
print('log Welch t-test', stats.ttest_ind(log_on2, log_off2, equal_var=False))
