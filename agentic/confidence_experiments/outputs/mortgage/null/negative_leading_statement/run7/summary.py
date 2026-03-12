import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.formula.api as smf


df = pd.read_csv('mortgage.csv')

# drop missing female/accept
sub = df[['accept','female']].dropna()

# counts
counts = sub.groupby('female')['accept'].agg(['sum','count'])
# success=accept
success = counts['sum'].values
nobs = counts['count'].values

print('counts by female')
print(counts)

# z-test for difference in proportions
stat, pval = proportions_ztest(success, nobs)
print('ztest stat', stat, 'pval', pval)

# simple logit
model_simple = smf.logit('accept ~ female', data=sub).fit(disp=False)
print(model_simple.summary())

# full logit with covariates
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
sub_full = df[['accept','female'] + covariates].dropna()
model_full = smf.logit('accept ~ female + ' + ' + '.join(covariates), data=sub_full).fit(disp=False)
print(model_full.summary())

# odds ratio and CI for female
params = model_full.params
conf = model_full.conf_int()

or_female = float(np.exp(params['female']))
conf_or = np.exp(conf.loc['female']).tolist()
print('female OR', or_female, 'CI', conf_or)

# average marginal effect
try:
    ame = model_full.get_margeff(at='overall', method='dydx')
    print(ame.summary())
except Exception as e:
    print('margeff error', e)
