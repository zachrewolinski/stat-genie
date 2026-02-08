import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
# feature2: frequency of affairs (numeric coding)
# feature6: children yes/no

# Clean
_df = _df.copy()
_df['has_children'] = _df['feature6'].str.lower().map({'yes':1,'no':0})

# Basic group stats
res = {}
for grp_val, grp_name in [(0,'no_children'), (1,'children')]:
    sub = _df[_df['has_children']==grp_val]
    res[grp_name] = {
        'n': len(sub),
        'mean_affairs': sub['feature2'].mean(),
        'median_affairs': sub['feature2'].median(),
        'prop_any_affair': (sub['feature2']>0).mean(),
    }

# Two-sample tests
no = _df[_df['has_children']==0]['feature2']
ch = _df[_df['has_children']==1]['feature2']

# Welch t-test
_ttest = stats.ttest_ind(no, ch, equal_var=False)

# Mann-Whitney U (nonparametric)
_mwu = stats.mannwhitneyu(no, ch, alternative='two-sided')

# Proportion test for any affairs >0
p1 = (no>0).mean()
p2 = (ch>0).mean()

n1 = len(no)
n2 = len(ch)
# two-proportion z-test
p_pool = ((no>0).sum() + (ch>0).sum())/(n1+n2)
se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
if se == 0:
    z = np.nan
    pval_prop = np.nan
else:
    z = (p1 - p2)/se
    pval_prop = 2*(1-stats.norm.cdf(abs(z)))

# Effect sizes
# Cohen's d (using pooled SD)
mean_diff = no.mean() - ch.mean()

s1 = no.var(ddof=1)
s2 = ch.var(ddof=1)
sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2))
cohen_d = mean_diff/sp if sp != 0 else np.nan

# Cliff's delta for nonparametric effect size
# compute using ranks
# Efficient computation
no_vals = no.values
ch_vals = ch.values
# Use broadcasting with chunking to avoid huge memory? n=601 so fine
m = len(no_vals)
n = len(ch_vals)
# Count greater/less
# compute with numpy broadcasting
# For memory: m*n about <= 200k maybe; actually could be ~300k, ok

diff = no_vals[:, None] - ch_vals[None, :]
cliffs_delta = (np.sum(diff>0) - np.sum(diff<0)) / (m*n)

# Print results
print('Group stats:', res)
print('Welch t-test:', _ttest)
print('Mann-Whitney U:', _mwu)
print('Prop any affair: no_children', p1, 'children', p2, 'z', z, 'p', pval_prop)
print('Mean diff (no - children):', mean_diff)
print('Cohen d:', cohen_d)
print('Cliffs delta:', cliffs_delta)
