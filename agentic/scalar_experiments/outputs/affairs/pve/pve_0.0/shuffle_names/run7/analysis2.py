import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
df = pd.read_csv('affairs.csv')

# Map columns based on info.json descriptions
# 'religiousness' column is actually children yes/no
# 'age' column is actually affairs frequency (per description)
children_col = 'religiousness'
outcome_col = 'age'

# Clean
# ensure binary
print('children levels:', df[children_col].unique())

# group stats
summary = df.groupby(children_col)[outcome_col].agg(['count','mean','std','median'])
print('\nGroup summary:')
print(summary)

# Welch t-test
child_yes = df[df[children_col]=='yes'][outcome_col]
child_no = df[df[children_col]=='no'][outcome_col]

t_stat, p_val = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')
print('\nWelch t-test: t=%.4f, p=%.4g' % (t_stat, p_val))

# Effect size (Cohen's d with pooled SD)
# Use standard pooled sd for independent groups
n1, n2 = child_yes.size, child_no.size
s1, s2 = child_yes.std(ddof=1), child_no.std(ddof=1)
pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
cohen_d = (child_yes.mean() - child_no.mean()) / pooled_sd
print('Cohen d (yes - no): %.4f' % cohen_d)

# OLS regression with binary indicator
# encode yes=1, no=0
X = (df[children_col]=='yes').astype(int)
X = sm.add_constant(X)
model = sm.OLS(df[outcome_col], X).fit()
print('\nOLS summary:')
print(model.summary())

# Nonparametric test
u_stat, u_p = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
print('\nMann-Whitney U: U=%.4f, p=%.4g' % (u_stat, u_p))
