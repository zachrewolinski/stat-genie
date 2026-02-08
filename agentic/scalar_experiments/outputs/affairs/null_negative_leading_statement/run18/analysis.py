import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# normalize column names if needed
print('columns', df.columns.tolist())

# basic checks
print('rows', len(df))

# Create binary any affair
df['any_affair'] = (df['affairs'] > 0).astype(int)

# children as binary: yes=1, no=0
# Some datasets might have yes/no strings
child_map = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, True: 1, False: 0, 1: 1, 0: 0}

df['children_bin'] = df['children'].map(child_map)

if df['children_bin'].isna().any():
    # if something unexpected, try to coerce
    df['children_bin'] = df['children'].astype(str).str.lower().map({'yes': 1, 'no': 0})

print('children_bin value counts')
print(df['children_bin'].value_counts(dropna=False))

# Group stats
summary = df.groupby('children_bin')['affairs'].agg(['count','mean','median','std'])
print('affairs by children_bin')
print(summary)

summary_any = df.groupby('children_bin')['any_affair'].agg(['mean','count'])
print('any_affair by children_bin')
print(summary_any)

# t-test (Welch) on affairs
from scipy import stats

a = df.loc[df['children_bin'] == 1, 'affairs']
b = df.loc[df['children_bin'] == 0, 'affairs']

# remove missing
an = a.dropna()
bn = b.dropna()

if len(an) > 1 and len(bn) > 1:
    tstat, pval = stats.ttest_ind(an, bn, equal_var=False)
    print('t_test_affairs children yes vs no: t=', tstat, 'p=', pval)

# Mann-Whitney
if len(an) > 1 and len(bn) > 1:
    ustat, pval_u = stats.mannwhitneyu(an, bn, alternative='two-sided')
    print('mw_affairs children yes vs no: U=', ustat, 'p=', pval_u)

# Logistic regression for any affair
# controls: gender, age, yearsmarried, religiousness, education, occupation, rating
# C() for categorical

formula_logit = 'any_affair ~ children_bin + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
logit_model = smf.logit(formula_logit, data=df).fit(disp=False)
print('logit summary children_bin coef/p')
print(logit_model.params['children_bin'], logit_model.pvalues['children_bin'])

# OLS on affairs
formula_ols = 'affairs ~ children_bin + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
ols_model = smf.ols(formula_ols, data=df).fit()
print('ols summary children_bin coef/p')
print(ols_model.params['children_bin'], ols_model.pvalues['children_bin'])

# Poisson regression for affairs counts
poisson_model = smf.glm(formula_ols, data=df, family=sm.families.Poisson()).fit()
print('poisson summary children_bin coef/p')
print(poisson_model.params['children_bin'], poisson_model.pvalues['children_bin'])

# Also compute predicted effect in logistic as odds ratio
odds_ratio = np.exp(logit_model.params['children_bin'])
print('logit odds_ratio children_bin', odds_ratio)

# Basic effect size for mean difference
if len(an) > 1 and len(bn) > 1:
    diff = an.mean() - bn.mean()
    # Cohen's d
    s1 = an.var(ddof=1)
    s2 = bn.var(ddof=1)
    n1 = len(an)
    n2 = len(bn)
    pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2))
    d = diff/pooled if pooled > 0 else np.nan
    print('mean diff (children yes - no):', diff)
    print("cohen's d:", d)

