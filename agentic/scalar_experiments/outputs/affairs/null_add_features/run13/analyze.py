import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('affairs.csv')

# normalize column names? assume as in info
# create binary affair indicator

df['has_affair'] = (df['affairs'] > 0).astype(int)

df['children_yes'] = (df['children'].str.lower() == 'yes').astype(int)

# group stats

group = df.groupby('children_yes')

summary = group['affairs'].agg(['count','mean','std'])
summary_affair = group['has_affair'].agg(['mean','count'])

# t-test on affairs count

child_yes = df.loc[df['children_yes']==1, 'affairs']
child_no = df.loc[df['children_yes']==0, 'affairs']

ttest = stats.ttest_ind(child_yes, child_no, equal_var=False)

# Mann-Whitney
mw = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')

# difference in proportions for has_affair
p_yes = df.loc[df['children_yes']==1, 'has_affair'].mean()
p_no = df.loc[df['children_yes']==0, 'has_affair'].mean()

# z-test for proportions
n_yes = df.loc[df['children_yes']==1, 'has_affair'].count()
n_no = df.loc[df['children_yes']==0, 'has_affair'].count()

p_pool = (p_yes*n_yes + p_no*n_no) / (n_yes+n_no)
se = np.sqrt(p_pool*(1-p_pool)*(1/n_yes + 1/n_no))
if se > 0:
    z = (p_yes - p_no)/se
    pval_prop = 2*(1-stats.norm.cdf(abs(z)))
else:
    z = np.nan
    pval_prop = np.nan

# logistic regression adjusted
# select covariates if exist
covars = ['children_yes','age','yearsmarried','gender','religiousness','education','occupation','rating']
# encode gender
X = df[covars].copy()
# one-hot for gender if categorical
X = pd.get_dummies(X, columns=['gender'], drop_first=True)
X = sm.add_constant(X)

y = df['has_affair']

logit = sm.Logit(y, X).fit(disp=False)

# linear regression on affairs count (OLS) adjusted
ols = sm.OLS(df['affairs'], X).fit()

print('Counts by children_yes (1=yes):')
print(summary)
print('\nProportion any affair:')
print(summary_affair)
print('\nMean affairs yes/no:', child_yes.mean(), child_no.mean())
print('Difference mean (yes - no):', child_yes.mean() - child_no.mean())
print('t-test:', ttest)
print('mannwhitney:', mw)
print('\nProportion any affair yes/no:', p_yes, p_no)
print('Diff proportion (yes - no):', p_yes - p_no)
print('z-test:', z, 'p', pval_prop)

print('\nLogit children_yes coef:', logit.params['children_yes'])
print('Logit children_yes OR:', np.exp(logit.params['children_yes']))
print('Logit pvalue:', logit.pvalues['children_yes'])

print('\nOLS children_yes coef:', ols.params['children_yes'])
print('OLS pvalue:', ols.pvalues['children_yes'])

