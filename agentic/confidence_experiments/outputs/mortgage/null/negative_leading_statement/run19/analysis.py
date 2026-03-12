import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Basic counts
n = len(df)

# Acceptance rate by gender
accept_rate = df.groupby('female')['accept'].mean()
counts = df['female'].value_counts().sort_index()

# 2x2 contingency for accept/deny by female
contingency = pd.crosstab(df['female'], df['accept'])
chi2, pval, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression: accept on female with controls
# Use available covariates (exclude accept/deny duplicates)
# We'll treat female as main predictor

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

formula = 'accept ~ female + ' + ' + '.join(covariates)

model = smf.logit(formula=formula, data=df).fit(disp=False)

# Extract female coefficient, OR, CI
coef = model.params['female']
se = model.bse['female']
z = coef / se
p = model.pvalues['female']

# Odds ratio and 95% CI
or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

# Also unadjusted logistic regression
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=False)
coef_u = model_unadj.params['female']
se_u = model_unadj.bse['female']
ora_u = np.exp(coef_u)
ci_low_u = np.exp(coef_u - 1.96*se_u)
ci_high_u = np.exp(coef_u + 1.96*se_u)
p_u = model_unadj.pvalues['female']

print('n', n)
print('counts female 0/1', counts.to_dict())
print('accept_rate by female', accept_rate.to_dict())
print('chi2', chi2, 'p', pval)
print('unadj OR', ora_u, '95% CI', (ci_low_u, ci_high_u), 'p', p_u)
print('adj OR', or_val, '95% CI', (ci_low, ci_high), 'p', p)

# compute marginal effect? use get_margeff
try:
    margeff = model.get_margeff(at='overall')
    me = margeff.summary_frame().loc['female']
    print('adj marginal effect (dy/dx) for female', me.to_dict())
except Exception as e:
    print('marginal effect failed', e)
