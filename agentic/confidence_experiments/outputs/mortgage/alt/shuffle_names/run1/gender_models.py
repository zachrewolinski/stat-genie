import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('mortgage.csv')

approval = _df['deny']
gender = _df['denied_PMI']

# candidate covariates based on metadata descriptions
cov_sets = {
    'basic': ['consumer_credit', 'mortgage_credit', 'Unnamed: 0', 'housing_expense_ratio', 'PI_ratio', 'married', 'black'],
    'plus_married_selfemp': ['consumer_credit', 'mortgage_credit', 'Unnamed: 0', 'housing_expense_ratio', 'PI_ratio', 'married', 'black', 'accept', 'loan_to_value'],
}

for name, covs in cov_sets.items():
    X = _df[covs].copy()
    X['gender'] = gender
    X = sm.add_constant(X)
    model = sm.Logit(approval, X, missing='drop')
    res = model.fit(disp=False, maxiter=200)
    coef = res.params['gender']
    p = res.pvalues['gender']
    or_val = np.exp(coef)
    print(name, 'gender coef', coef, 'OR', or_val, 'p', p)

# Linear probability model with robust SEs
covs = cov_sets['basic']
X = _df[covs].copy()
X['gender'] = gender
X = sm.add_constant(X)
model_lpm = sm.OLS(approval, X, missing='drop')
res_lpm = model_lpm.fit(cov_type='HC3')
print('LPM gender coef', res_lpm.params['gender'], 'p', res_lpm.pvalues['gender'])

