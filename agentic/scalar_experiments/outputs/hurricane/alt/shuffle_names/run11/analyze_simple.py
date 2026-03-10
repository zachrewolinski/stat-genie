import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('hurricane.csv')
_df2 = _df.rename(columns={
    'category': 'femininity_rating',
    'name': 'deaths',
    'masfem_mturk': 'fem_binary',
    'ind': 'femininity_mturk'
})

_df2['log_deaths'] = np.log1p(_df2['deaths'])

# simple regression
X = sm.add_constant(_df2['femininity_rating'])
y = _df2['log_deaths']
model = sm.OLS(y, X).fit(cov_type='HC3')
print(model.summary())

# binary
Xb = sm.add_constant(_df2['fem_binary'])
model_b = sm.OLS(y, Xb).fit(cov_type='HC3')
print('\n', model_b.summary())
