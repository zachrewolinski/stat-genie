import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
affairs = _df['feature2']
children = _df['feature6']

# Basic groups
with_children = affairs[children == 'yes']
without_children = affairs[children == 'no']

# Means and proportions
mean_with = with_children.mean()
mean_without = without_children.mean()

prop_with = (with_children > 0).mean()
prop_without = (without_children > 0).mean()

# Difference
diff_mean = mean_with - mean_without
ratio_mean = mean_with / mean_without if mean_without != 0 else np.nan

# Two-sample t-test (unequal variances)
_ttest = stats.ttest_ind(with_children, without_children, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)
_mwu = stats.mannwhitneyu(with_children, without_children, alternative='two-sided')

# Effect size: Cohen's d (pooled SD)
# Note: use pooled SD with unequal n
n1 = len(with_children)
n2 = len(without_children)
var1 = with_children.var(ddof=1)
var2 = without_children.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohen_d = (mean_with - mean_without) / pooled_sd if pooled_sd != 0 else np.nan

# Difference in proportions test (two-proportion z)
# Use normal approximation
p1 = prop_with
p2 = prop_without
p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 and p_pool < 1 else np.nan
z = (p1 - p2) / se if se and se != 0 else np.nan
pval_prop = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan

print('n_with_children', n1)
print('n_without_children', n2)
print('mean_with_children', mean_with)
print('mean_without_children', mean_without)
print('diff_mean_with_minus_without', diff_mean)
print('ratio_mean_with_div_without', ratio_mean)
print('prop_any_affair_with_children', prop_with)
print('prop_any_affair_without_children', prop_without)
print('diff_prop_with_minus_without', p1 - p2)
print('ttest_stat', _ttest.statistic)
print('ttest_pvalue', _ttest.pvalue)
print('mwu_stat', _mwu.statistic)
print('mwu_pvalue', _mwu.pvalue)
print('cohen_d', cohen_d)
print('prop_z', z)
print('prop_pvalue', pval_prop)
