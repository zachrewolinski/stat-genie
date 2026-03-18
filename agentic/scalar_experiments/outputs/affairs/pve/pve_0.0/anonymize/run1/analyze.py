import json
import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Map children yes/no
children = df['feature6']
# outcome
affairs = df['feature2']

# group stats
summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])
print(summary)

# t-test (Welch)
vals_yes = affairs[children == 'yes']
vals_no = affairs[children == 'no']

t_stat, p_val = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')
print('Welch t-test: t=%.4f p=%.6f' % (t_stat, p_val))

# Mann-Whitney U (nonparam)
try:
    u_stat, p_u = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
    print('Mann-Whitney: U=%.4f p=%.6f' % (u_stat, p_u))
except Exception as e:
    print('Mann-Whitney error', e)

# Any affair indicator
any_affair = (affairs > 0).astype(int)
ct = pd.crosstab(children, any_affair)
print('crosstab children x any_affair')
print(ct)

# chi-square test
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print('Chi-square: chi2=%.4f p=%.6f' % (chi2, p_chi))

# proportions
prop_yes = any_affair[children=='yes'].mean()
prop_no = any_affair[children=='no'].mean()
print('Proportion any affair: yes=%.4f no=%.4f' % (prop_yes, prop_no))

# effect size: Cohen's d
mean_yes, mean_no = vals_yes.mean(), vals_no.mean()
std_yes, std_no = vals_yes.std(ddof=1), vals_no.std(ddof=1)
# pooled sd
n_yes, n_no = vals_yes.shape[0], vals_no.shape[0]
pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes+n_no-2))
cohen_d = (mean_yes - mean_no) / pooled_sd
print('Means yes=%.4f no=%.4f cohen_d=%.4f' % (mean_yes, mean_no, cohen_d))
