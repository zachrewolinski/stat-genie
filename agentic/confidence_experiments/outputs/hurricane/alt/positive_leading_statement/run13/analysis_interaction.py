import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DF = pd.read_csv('hurricane.csv')
DF['log_deaths'] = np.log1p(DF['alldeaths'])

# Interaction models: use wind and/or min as severity
models = {}

models['wind_interaction'] = smf.ols(
    'log_deaths ~ masfem * wind + min + category + year', data=DF
).fit(cov_type='HC3')

models['min_interaction'] = smf.ols(
    'log_deaths ~ masfem * min + wind + category + year', data=DF
).fit(cov_type='HC3')

models['category_interaction'] = smf.ols(
    'log_deaths ~ masfem * category + wind + min + year', data=DF
).fit(cov_type='HC3')

for name, model in models.items():
    print(f'\nModel: {name}')
    print(model.summary().tables[1])
