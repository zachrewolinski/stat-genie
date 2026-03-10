import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# load data
info = json.load(open('info.json'))
df = pd.read_csv('mortgage.csv')

# identify gender and approval columns based on metadata descriptions
fields = info['data_desc']['fields']

gender_col = None
approval_col = None
for f in fields:
    desc = (f.get('properties', {}).get('description') or '').lower()
    col = f['column']
    if 'female' in desc and 'male' in desc:
        gender_col = col
    if 'accepted' in desc and 'denied' in desc:
        approval_col = col

print('gender_col', gender_col, 'approval_col', approval_col)

# fallback in case metadata not found
if gender_col is None:
    gender_col = 'female'
if approval_col is None:
    approval_col = 'deny'

# keep only 0/1 values
sub = df[[gender_col, approval_col]].dropna()
sub = sub[(sub[gender_col].isin([0,1])) & (sub[approval_col].isin([0,1]))]

# contingency table
ct = pd.crosstab(sub[gender_col], sub[approval_col])
print('contingency table (gender x approval):\n', ct)

# rates
rates = ct.div(ct.sum(axis=1), axis=0)
print('rates:\n', rates)

# chi-square
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p_chi)

# difference in proportions and CI
p1 = rates.loc[1, 1] if 1 in rates.index else np.nan
p0 = rates.loc[0, 1] if 0 in rates.index else np.nan
n1 = ct.loc[1].sum() if 1 in ct.index else np.nan
n0 = ct.loc[0].sum() if 0 in ct.index else np.nan

se = np.sqrt((p1*(1-p1)/n1) + (p0*(1-p0)/n0)) if n1>0 and n0>0 else np.nan
z = (p1-p0)/se if se and se>0 else np.nan
p_z = 2*(1-stats.norm.cdf(abs(z))) if se and se>0 else np.nan
ci_low = (p1-p0) - 1.96*se if se and se>0 else np.nan
ci_high = (p1-p0) + 1.96*se if se and se>0 else np.nan
print('p1', p1, 'p0', p0, 'diff', p1-p0, 'z', z, 'p_z', p_z, 'ci', (ci_low, ci_high))

# logistic regression with gender only
X = sm.add_constant(sub[gender_col])
model = sm.Logit(sub[approval_col], X)
res = model.fit(disp=False)
print(res.summary())

# logistic regression with gender + other covariates (exclude target & gender)
cols = [c for c in df.columns if c not in [gender_col, approval_col]]
X2 = df[cols].copy()
X2 = X2.select_dtypes(include=[np.number])

# drop columns with too many unique values (likely IDs)
for c in list(X2.columns):
    if X2[c].nunique() > 0.9 * len(X2):
        X2 = X2.drop(columns=[c])

# build combined dataset and drop missing
X2 = pd.concat([df[[gender_col, approval_col]], X2], axis=1)
X2 = X2.dropna()
X2 = X2[(X2[gender_col].isin([0,1])) & (X2[approval_col].isin([0,1]))]

# design matrix
Y2 = X2[approval_col]
X2_mat = sm.add_constant(X2.drop(columns=[approval_col]))

# drop columns with zero variance
for c in list(X2_mat.columns):
    if X2_mat[c].nunique() <= 1:
        X2_mat = X2_mat.drop(columns=[c])

model2 = sm.Logit(Y2, X2_mat)
res2 = model2.fit(disp=False, maxiter=200)
print(res2.summary())

out = {
    'gender_col': gender_col,
    'approval_col': approval_col,
    'ct': ct.to_dict(),
    'rates': rates.to_dict(),
    'chi2_p': p_chi,
    'diff': p1-p0,
    'diff_ci': [ci_low, ci_high],
    'z_p': p_z,
    'logit_gender_coef': res.params[gender_col],
    'logit_gender_p': res.pvalues[gender_col],
}

if gender_col in res2.params.index:
    out['adj_gender_coef'] = res2.params[gender_col]
    out['adj_gender_p'] = res2.pvalues[gender_col]
    out['adj_n'] = int(res2.nobs)

print('OUT', out)
