import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm

# Load data

df = pd.read_csv('hurricane.csv')
num_cols = ['masfem','alldeaths','wind','min','category','year']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

X = df[['masfem','wind','min','category','year']]
X = sm.add_constant(X)

y = df['alldeaths']

nb2 = dm.NegativeBinomial(y, X)
res = nb2.fit(disp=False, maxiter=200)

# apply robust cov
res._get_robustcov_results(cov_type='HC3')

out = {
    'coef': float(res.params['masfem']),
    'se': float(res.bse['masfem']),
    'p': float(res.pvalues['masfem']),
    'alpha': float(res.params['alpha'])
}

print(out)
