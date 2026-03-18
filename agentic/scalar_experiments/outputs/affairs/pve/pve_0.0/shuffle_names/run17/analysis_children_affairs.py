import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load

df = pd.read_csv('affairs.csv')

# Map children indicator: column 'religiousness' yes/no per metadata
children_col = 'religiousness'
# Outcome (affairs frequency): column 'age' per metadata
outcome_col = 'age'

# Clean: drop missing
sub = df[[children_col, outcome_col]].dropna()

# Encode children: yes=1, no=0
child_map = {'yes': 1, 'no': 0}
sub['children'] = sub[children_col].map(child_map)

# Group stats
means = sub.groupby('children')[outcome_col].agg(['mean','std','count'])
print(means)

# t-test
vals_yes = sub[sub['children']==1][outcome_col]
vals_no = sub[sub['children']==0][outcome_col]

t_stat, p_val = stats.ttest_ind(vals_yes, vals_no, equal_var=False)
print('t_stat', t_stat, 'p_val', p_val)

# effect size (Cohen d)
mean_diff = vals_yes.mean() - vals_no.mean()

# pooled sd (for Cohen d)
# Use weighted sd
n1, n0 = len(vals_yes), len(vals_no)
var1, var0 = vals_yes.var(ddof=1), vals_no.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohen_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan
print('mean_diff', mean_diff, 'cohen_d', cohen_d)

# Non-parametric test (Mann-Whitney)
U_stat, p_mw = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
print('Mann-Whitney p', p_mw)

# Simple regression
X = sm.add_constant(sub['children'])
model = sm.OLS(sub[outcome_col], X).fit()
print(model.summary())

