import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

with open('info.json', 'r') as f:
    info = json.load(f)

fields = info['data_desc']['fields']

# Identify columns by description text
sex_col = None
outcome_col = None
for field in fields:
    desc = (field.get('properties', {}).get('description') or '').lower()
    if 'female' in desc:
        sex_col = field['column']
    if 'mortgage application was denied' in desc:
        outcome_col = field['column']

print('sex_col', sex_col)
print('outcome_col', outcome_col)

if sex_col is None or outcome_col is None:
    raise SystemExit('Required columns not found')

# Load data

df = pd.read_csv('mortgage.csv')

# Drop missing in sex or outcome
sub = df[[sex_col, outcome_col]].dropna()

# outcome_col is denial indicator (1=denied); create approval
sub['approved'] = 1 - sub[outcome_col]

# crosstab
ct = pd.crosstab(sub[sex_col], sub['approved'])
print('crosstab sex vs approved')
print(ct)

# rates
female_rate = ct.loc[1, 1] / ct.loc[1].sum()
male_rate = ct.loc[0, 1] / ct.loc[0].sum()
print('female approval rate', female_rate)
print('male approval rate', male_rate)
print('diff female - male', female_rate - male_rate)

# chi-square
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p_chi)

# proportion z-test
count = np.array([ct.loc[1,1], ct.loc[0,1]])
nobs = np.array([ct.loc[1].sum(), ct.loc[0].sum()])
stat, p_z = proportions_ztest(count, nobs)
print('ztest', stat, 'p', p_z)

# Logistic regression unadjusted
X = sm.add_constant(sub[sex_col])
model = sm.Logit(sub['approved'], X).fit(disp=False)
print(model.summary())

# Adjusted model with all other columns except outcome and sex
cov_cols = [c for c in df.columns if c not in {sex_col, outcome_col}]
sub2 = df[[sex_col, outcome_col] + cov_cols].dropna()
sub2['approved'] = 1 - sub2[outcome_col]
X2 = sm.add_constant(sub2[cov_cols + [sex_col]])
model2 = sm.Logit(sub2['approved'], X2).fit(disp=False, maxiter=200)
print(model2.summary())

# effect sizes
or_unadj = np.exp(model.params[sex_col])
or_adj = np.exp(model2.params[sex_col])
print('unadj OR', or_unadj, 'p', model.pvalues[sex_col])
print('adj OR', or_adj, 'p', model2.pvalues[sex_col])

# sample sizes
print('n unadjusted', len(sub))
print('n adjusted', len(sub2))
