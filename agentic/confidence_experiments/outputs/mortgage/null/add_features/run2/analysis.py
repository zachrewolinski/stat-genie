import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load dataset
_df = pd.read_csv('mortgage.csv')

print('rows', len(_df))
print('columns', _df.columns.tolist())

# Focus on gender and approval
# Determine outcome variable: accept or deny
# We'll use accept if present, else use deny.

# Basic cleaning: ensure numeric and drop missing

# compute basic approval rate by gender

def safe_col(name):
    return name if name in _df.columns else None

female_col = safe_col('female')
accept_col = safe_col('accept')
deny_col = safe_col('deny')

print('female col', female_col, 'accept col', accept_col, 'deny col', deny_col)

if female_col is None or (accept_col is None and deny_col is None):
    raise SystemExit('Required columns missing')

# Use accept; if not, use deny as inverse
if accept_col is None:
    _df['accept'] = 1 - _df[deny_col]
    accept_col = 'accept'

# Make sure binary

# compute counts
subset = _df[[female_col, accept_col]].dropna()

# ensure 0/1

print('accept unique', subset[accept_col].unique())
print('female unique', subset[female_col].unique())

# Crosstab
ct = pd.crosstab(subset[female_col], subset[accept_col])
print('crosstab')
print(ct)

# approval rates
rates = ct.div(ct.sum(axis=1), axis=0)
print('approval rates by female=0/1')
print(rates)

# chi-square test for independence
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p, 'dof', dof)

# Two-proportion z-test for approval rates
# female=1 vs female=0
n_f = ct.loc[1].sum() if 1 in ct.index else None
n_m = ct.loc[0].sum() if 0 in ct.index else None
if n_f and n_m:
    approve_f = ct.loc[1, 1] if 1 in ct.columns else 0
    approve_m = ct.loc[0, 1] if 1 in ct.columns else 0
    prop_f = approve_f / n_f
    prop_m = approve_m / n_m
    # z-test for difference in proportions
    p_pool = (approve_f + approve_m) / (n_f + n_m)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_f + 1/n_m))
    z = (prop_f - prop_m) / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
    print('prop_f', prop_f, 'prop_m', prop_m, 'diff', prop_f - prop_m, 'z', z, 'p_z', p_z)

# Logistic regression controlling for other covariates
# Choose some relevant covariates if present
covariates = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]

covariates = [c for c in covariates if c in _df.columns]

model_cols = [female_col, accept_col] + covariates
model_df = _df[model_cols].dropna()

# Add constant
X = model_df[[female_col] + covariates]
X = sm.add_constant(X)

# Outcome
y = model_df[accept_col]

# Fit logistic regression
try:
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)
    print('logit nobs', result.nobs)
    print(result.summary2().tables[1].loc[female_col])
    # Odds ratio
    coef = result.params[female_col]
    se = result.bse[female_col]
    or_val = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    print('female odds ratio', or_val, '95% CI', (ci_low, ci_high))
except Exception as e:
    print('logit error', e)

