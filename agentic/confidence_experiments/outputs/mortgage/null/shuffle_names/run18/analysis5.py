import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
with open('info.json', 'r') as f:
    info = json.load(f)

df = pd.read_csv('mortgage.csv')

# Identify gender and approval columns based on descriptions
fields = info['data_desc']['fields']

def find_col(keyword):
    for field in fields:
        desc = (field.get('properties', {}).get('description') or '').lower()
        if keyword in desc:
            return field['column'], desc
    return None, None

# gender column (female)
sex_col, sex_desc = find_col('female')
# approval column (accepted)
approve_col, approve_desc = find_col('accepted')

print('sex_col', sex_col, sex_desc)
print('approve_col', approve_col, approve_desc)

# also find denied column if needed
print('denied column maybe:', find_col('denied'))

# Basic checks
if sex_col is None or approve_col is None:
    raise SystemExit('Required columns not found')

# Ensure binary
print('sex unique', sorted(df[sex_col].unique()))
print('approve unique', sorted(df[approve_col].unique()))

# Compute approval rates by gender
ct = pd.crosstab(df[sex_col], df[approve_col])
print('crosstab sex vs approve')
print(ct)

# Proportions
female_rate = ct.loc[1, 1] / ct.loc[1].sum()
male_rate = ct.loc[0, 1] / ct.loc[0].sum()
print('female approval rate', female_rate)
print('male approval rate', male_rate)
print('diff female - male', female_rate - male_rate)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Two-proportion z-test (approx)
from statsmodels.stats.proportion import proportions_ztest
count = np.array([ct.loc[1,1], ct.loc[0,1]])
nobs = np.array([ct.loc[1].sum(), ct.loc[0].sum()])
stat, p_z = proportions_ztest(count, nobs)
print('ztest', stat, 'p', p_z)

# Logistic regression unadjusted
X = sm.add_constant(df[sex_col])
model = sm.Logit(df[approve_col], X).fit(disp=False)
print(model.summary())

# Logistic regression adjusted: use all other columns except outcome
cov_cols = [c for c in df.columns if c not in {approve_col, sex_col}]
X2 = sm.add_constant(df[cov_cols])
model2 = sm.Logit(df[approve_col], X2).fit(disp=False, maxiter=200)
print(model2.summary())

# Odds ratio for sex in adjusted model if present
if sex_col in model2.params:
    or_adj = np.exp(model2.params[sex_col])
    print('adj OR', or_adj, 'p', model2.pvalues[sex_col])

