import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

# Define variables

df['children'] = (df['feature6'].str.lower() == 'yes').astype(int)
df['male'] = (df['feature3'].str.lower() == 'male').astype(int)
df['any_affair'] = (df['feature2'] > 0).astype(int)

# Group stats

groups = df.groupby('children')['feature2']
summary = groups.agg(['count','mean','median','std'])
summary['prop_any_affair'] = df.groupby('children')['any_affair'].mean()

# Welch t-test

g0 = df[df['children'] == 0]['feature2']
g1 = df[df['children'] == 1]['feature2']

ttest = stats.ttest_ind(g0, g1, equal_var=False)

# Mann-Whitney U test

mw = stats.mannwhitneyu(g0, g1, alternative='two-sided')

# Cohen's d (unequal n, pooled SD)

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)

cd = cohens_d(g0, g1)

# OLS regression

X = df[['children','male','feature4','feature5','feature7','feature8','feature9','feature10']].copy()
X = sm.add_constant(X)
ols = sm.OLS(df['feature2'], X).fit(cov_type='HC3')

# Logistic regression for any affair

logit = sm.Logit(df['any_affair'], X).fit(disp=False)

# Extract key results

def coef_summary(model, var):
    coef = model.params[var]
    se = model.bse[var]
    p = model.pvalues[var]
    return coef, se, p

ols_children = coef_summary(ols, 'children')
logit_children = coef_summary(logit, 'children')

# Odds ratio for logit

odds_ratio = np.exp(logit_children[0])

results = {
    'summary_by_children': summary.to_dict(),
    'ttest': {'stat': ttest.statistic, 'pvalue': ttest.pvalue},
    'mannwhitney': {'stat': mw.statistic, 'pvalue': mw.pvalue},
    'cohens_d_children0_minus_children1': cd,
    'ols_children': {'coef': ols_children[0], 'se': ols_children[1], 'pvalue': ols_children[2]},
    'logit_children': {'coef': logit_children[0], 'se': logit_children[1], 'pvalue': logit_children[2], 'odds_ratio': odds_ratio},
}

print(json.dumps(results, indent=2))
