import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Map columns
affairs = df['feature2']
children = df['feature6']

# Basic group stats
summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# Proportion with any affairs
any_affair = (affairs > 0).astype(int)
prop = df.groupby('feature6')['feature2'].apply(lambda s: (s>0).mean())

# t-test (Welch)
child_yes = affairs[children=='yes']
child_no = affairs[children=='no']

tstat, pval = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except Exception as e:
    u_stat, u_p = np.nan, np.nan

# Difference in proportions (affair>0) with z-test
from statsmodels.stats.proportion import proportions_ztest

count = np.array([(child_yes>0).sum(), (child_no>0).sum()])
obs = np.array([child_yes.size, child_no.size])

z_stat, z_p = proportions_ztest(count, obs, alternative='smaller')  # test yes < no

# also compute difference in means and Cohen's d
mean_diff = child_yes.mean() - child_no.mean()

# Cohen's d (pooled)
pooled_std = np.sqrt(((child_yes.var(ddof=1) + child_no.var(ddof=1))/2))
cohen_d = mean_diff / pooled_std if pooled_std>0 else np.nan

print('Summary by children:')
print(summary)
print('\nProportion with any affairs (>0):')
print(prop)
print('\nWelch t-test: t=%.4f p=%.4g' % (tstat, pval))
print('Mann-Whitney U: U=%.4f p=%.4g' % (u_stat, u_p))
print('Proportion z-test (yes < no): z=%.4f p=%.4g' % (z_stat, z_p))
print('Mean diff (yes-no)=%.4f, Cohen d=%.4f' % (mean_diff, cohen_d))
