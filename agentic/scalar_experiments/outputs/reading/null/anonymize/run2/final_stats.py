import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Derived reading speed (words per minute) using reading time excluding scrolling
_df['derived_wpm'] = _df['feature7'] / (_df['feature5'] / 60000.0)

# Subset to dyslexic participants
_df = _df.replace([np.inf, -np.inf], np.nan)
_df = _df[_df['feature17'] == 1]

# group by reader view
rv = _df[_df['feature3'] == 1]['derived_wpm'].dropna()
no_rv = _df[_df['feature3'] == 0]['derived_wpm'].dropna()

print('derived_wpm dyslexic n', len(rv), len(no_rv))
print('means', rv.mean(), no_rv.mean())
print('medians', rv.median(), no_rv.median())
print('std', rv.std(ddof=1), no_rv.std(ddof=1))

# Welch t-test
welch = stats.ttest_ind(rv, no_rv, equal_var=False, nan_policy='omit')
print('welch t', welch)

# Mann-Whitney
mw = stats.mannwhitneyu(rv, no_rv, alternative='two-sided')
print('mannwhitney', mw)

# Cohen's d
mean_diff = rv.mean() - no_rv.mean()
s1 = rv.var(ddof=1)
s2 = no_rv.var(ddof=1)
n1 = len(rv)
n2 = len(no_rv)
pooled_sd = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
print('cohens_d', d)

# Welch 95% CI for mean difference
se = np.sqrt(s1/n1 + s2/n2)
# Welch-Satterthwaite df
if n1 > 1 and n2 > 1:
    df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
    t_crit = stats.t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    print('mean_diff', mean_diff)
    print('ci', ci_low, ci_high)

# Also compute feature20 comparison as alternate speed proxy
rv2 = _df[_df['feature3'] == 1]['feature20'].dropna()
no_rv2 = _df[_df['feature3'] == 0]['feature20'].dropna()
print('feature20 dyslexic n', len(rv2), len(no_rv2))
print('feature20 means', rv2.mean(), no_rv2.mean())
print('feature20 medians', rv2.median(), no_rv2.median())
welch2 = stats.ttest_ind(rv2, no_rv2, equal_var=False, nan_policy='omit')
print('feature20 welch t', welch2)

