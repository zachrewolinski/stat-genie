import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Map columns
# feature2: affairs frequency
# feature6: children yes/no

df = df.copy()

# Clean

df['children'] = df['feature6'].astype(str).str.lower().map({'yes':1,'no':0})

# Summary stats

groups = df.groupby('children')

summary = groups['feature2'].agg(['count','mean','median','std'])

# Proportion with any affairs

df['any_affair'] = (df['feature2'] > 0).astype(int)
prop = groups['any_affair'].mean()

# Two-sample t-test (unequal var)

yes = df.loc[df['children']==1,'feature2']
no = df.loc[df['children']==0,'feature2']

ttest = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U
mwu = stats.mannwhitneyu(yes, no, alternative='two-sided')

# Logistic regression for any affair (unadjusted)
logit_unadj = smf.logit('any_affair ~ children', data=df).fit(disp=0)

# OLS for frequency (unadjusted)
ols_unadj = smf.ols('feature2 ~ children', data=df).fit()

# Adjusted models with basic covariates if present
# Use features: gender (feature3), age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10)

# Build formula with categorical for gender
formula_logit_adj = 'any_affair ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
formula_ols_adj = 'feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'

logit_adj = smf.logit(formula_logit_adj, data=df).fit(disp=0)
ols_adj = smf.ols(formula_ols_adj, data=df).fit()

# Extract key metrics

def coef_ci(model, term='children'):
    coef = model.params[term]
    se = model.bse[term]
    z = 1.96
    return coef, coef - z*se, coef + z*se

# For logit, compute odds ratio and CI
coef, lo, hi = coef_ci(logit_unadj)
or_unadj = np.exp(coef)
or_unadj_ci = (np.exp(lo), np.exp(hi))

coef_a, lo_a, hi_a = coef_ci(logit_adj)
or_adj = np.exp(coef_a)
or_adj_ci = (np.exp(lo_a), np.exp(hi_a))

# Write a brief report to stdout
print('Summary mean feature2 by children (0=no,1=yes):')
print(summary)
print('\nProportion any affair by children:')
print(prop)
print('\nT-test (feature2):', ttest)
print('Mann-Whitney U:', mwu)
print('\nOLS unadj coef children:', ols_unadj.params['children'], 'p', ols_unadj.pvalues['children'])
print('OLS adj coef children:', ols_adj.params['children'], 'p', ols_adj.pvalues['children'])
print('\nLogit unadj OR children:', or_unadj, 'CI', or_unadj_ci, 'p', logit_unadj.pvalues['children'])
print('Logit adj OR children:', or_adj, 'CI', or_adj_ci, 'p', logit_adj.pvalues['children'])
