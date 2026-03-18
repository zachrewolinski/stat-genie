import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
df = pd.read_csv('affairs.csv')

# Basic group stats for feature2 by children
children_col = 'feature6'
outcome_col = 'feature2'

df = df.copy()
# ensure children is categorical

df[children_col] = df[children_col].astype('category')

# group means, stds, counts
summary = df.groupby(children_col)[outcome_col].agg(['mean','std','count'])

# Cohen's d (yes - no)
if set(df[children_col].cat.categories) >= set(['yes','no']):
    yes = df[df[children_col]=='yes'][outcome_col].dropna()
    no = df[df[children_col]=='no'][outcome_col].dropna()
else:
    # fallback
    cats = df[children_col].cat.categories
    yes = df[df[children_col]==cats[0]][outcome_col].dropna()
    no = df[df[children_col]==cats[1]][outcome_col].dropna()

n1, n0 = len(yes), len(no)
mean1, mean0 = yes.mean(), no.mean()
std1, std0 = yes.std(ddof=1), no.std(ddof=1)
# pooled sd
sp = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2)/(n1+n0-2))
cohen_d = (mean1 - mean0)/sp if sp>0 else np.nan

# Welch t-test
ttest = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    mwu = stats.mannwhitneyu(yes, no, alternative='two-sided')
except ValueError:
    mwu = None

# OLS with controls (robust SE)
# C(feature6) is categorical; baseline 'no'
formula = 'feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
ols = smf.ols(formula, data=df).fit(cov_type='HC3')

# Logistic regression for any affairs (feature2 > 0)
# This is a rough proxy since feature2 is noisy/continuous

df['any_affair'] = (df[outcome_col] > 0).astype(int)
logit = smf.logit('any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(disp=False)

# Extract key stats
# For OLS, coefficient for children yes vs no
ols_params = ols.params
ols_pvalues = ols.pvalues
ols_ci = ols.conf_int(alpha=0.05)

# Find the term for C(feature6)[T.yes] (if exists)
child_term = None
for term in ols_params.index:
    if term.startswith('C(feature6)'):
        child_term = term
        break

logit_params = logit.params
logit_pvalues = logit.pvalues
logit_ci = logit.conf_int(alpha=0.05)

# Build output
print('GROUP SUMMARY')
print(summary)
print('\nMEAN DIFFERENCE (yes - no):', mean1-mean0)
print('COHEN D:', cohen_d)
print('\nWELCH T-TEST:', ttest)
if mwu:
    print('MANN-WHITNEY U:', mwu)

print('\nOLS HC3')
if child_term:
    print('Child term:', child_term)
    print('Coef:', ols_params[child_term])
    print('P-value:', ols_pvalues[child_term])
    print('95% CI:', tuple(ols_ci.loc[child_term]))

print('\nLOGIT')
if child_term:
    # logit uses same term
    if child_term in logit_params.index:
        print('Child term:', child_term)
        print('Coef (log-odds):', logit_params[child_term])
        print('P-value:', logit_pvalues[child_term])
        print('95% CI:', tuple(logit_ci.loc[child_term]))
        # odds ratio
        print('OR:', np.exp(logit_params[child_term]))
