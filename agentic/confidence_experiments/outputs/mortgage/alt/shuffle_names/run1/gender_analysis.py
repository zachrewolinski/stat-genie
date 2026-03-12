import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

_df = pd.read_csv('mortgage.csv')

approval = _df['deny']
# gender (1=female, 0=male) per metadata description
gender = _df['denied_PMI']

ct = pd.crosstab(gender, approval)
print('Crosstab gender (rows) x approval (cols):')
print(ct)

rate_female = ct.loc[1, 1] / ct.loc[1].sum() if 1 in ct.index else np.nan
rate_male = ct.loc[0, 1] / ct.loc[0].sum() if 0 in ct.index else np.nan
print('approval_rate_female', rate_female)
print('approval_rate_male', rate_male)
print('diff_female_minus_male', rate_female - rate_male)

chi2, p, dof, exp = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Unadjusted logistic regression: approval ~ gender
X = sm.add_constant(gender)
model = sm.Logit(approval, X, missing='drop')
res = model.fit(disp=False)
print('logit unadjusted params')
print(res.params)
print('pvalues')
print(res.pvalues)

# Adjusted logistic regression with covariates
covariates = [
    'consumer_credit',
    'mortgage_credit',
    'Unnamed: 0',
    'housing_expense_ratio',
    'PI_ratio',
    'loan_to_value',
    'married',
    'black',
    'accept',
]

X2 = _df[covariates].copy()
X2['gender'] = gender
X2 = sm.add_constant(X2)

model2 = sm.Logit(approval, X2, missing='drop')
res2 = model2.fit(disp=False, maxiter=200)
print('logit adjusted gender coef', res2.params['gender'], 'p', res2.pvalues['gender'])

