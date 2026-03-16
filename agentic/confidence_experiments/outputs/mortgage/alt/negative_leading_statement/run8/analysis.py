import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# Load data
_df = pd.read_csv('mortgage.csv')

# Standardize column names just in case
print('Columns:', _df.columns.tolist())

# Ensure binary columns are numeric
# female (1 female, 0 male). accept (1 accepted). deny (1 denied).

# Basic counts
n = len(_df)
print('Rows', n)

# Raw acceptance rates by gender
# female=1 vs 0
rates = _df.groupby('female')['accept'].mean()
counts = _df['female'].value_counts().sort_index()
print('Acceptance rates by female:', rates.to_dict())
print('Counts by female:', counts.to_dict())

# difference in proportions test
# success = accept
successes = _df.groupby('female')['accept'].sum().sort_index()
# order: 0,1
count = _df.groupby('female')['accept'].count().sort_index()
stat, pval = proportions_ztest(count=successes.values, nobs=count.values, alternative='two-sided')
print('Diff in accept rate z-test:', stat, pval)

# logistic regression: accept ~ female + covariates
# We'll include credit-related and financial covariates to adjust.
# Use available columns excluding outcome and index.

# define covariates
covariates = ['female','black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']

# ensure no missing
_df2 = _df[covariates + ['accept','deny']].dropna()

X = _df2[covariates]
X = sm.add_constant(X)
y = _df2['accept']

logit = sm.Logit(y, X)
res = logit.fit(disp=False)
print(res.summary())

# effect for female
coef = res.params['female']
se = res.bse['female']
p = res.pvalues['female']
# odds ratio
or_val = np.exp(coef)
# 95% CI
ci = res.conf_int().loc['female']
ci_or = np.exp(ci)
print('female coef', coef, 'se', se, 'p', p, 'OR', or_val, 'CI OR', ci_or.tolist())

# compute marginal effect (approx)
# We'll compute average marginal effect using statsmodels get_margeff
mfx = res.get_margeff(at='overall', method='dydx')
print(mfx.summary())

# Extract female marginal effect
mfx_df = mfx.summary_frame()
print('Female marginal effect', mfx_df.loc['female'].to_dict())

# Also check model with deny as outcome (should be inverse)
logit_deny = sm.Logit(_df2['deny'], X)
res_deny = logit_deny.fit(disp=False)
print('Deny female coef', res_deny.params['female'], 'p', res_deny.pvalues['female'])
