import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Center variables for interaction
_df['masfem_c'] = _df['masfem'] - _df['masfem'].mean()
_df['wind_c'] = _df['wind'] - _df['wind'].mean()
_df['min_c'] = _df['min'] - _df['min'].mean()

cols = ['log_deaths','masfem_c','wind_c','min_c','category','year']
_df = _df.dropna(subset=cols)

# Interaction with wind
Xw = _df[['masfem_c','wind_c']].copy()
Xw['masfem_wind'] = _df['masfem_c'] * _df['wind_c']
Xw = pd.concat([Xw, _df[['min_c','category','year']]], axis=1)
Xw = sm.add_constant(Xw)
model_w = sm.OLS(_df['log_deaths'], Xw).fit(cov_type='HC3')

# Interaction with min pressure (lower pressure more intense)
Xm = _df[['masfem_c','min_c']].copy()
Xm['masfem_min'] = _df['masfem_c'] * _df['min_c']
Xm = pd.concat([Xm, _df[['wind_c','category','year']]], axis=1)
Xm = sm.add_constant(Xm)
model_m = sm.OLS(_df['log_deaths'], Xm).fit(cov_type='HC3')

print('Interaction masfem x wind:')
print(model_w.summary().tables[1])
print('\nInteraction masfem x min:')
print(model_m.summary().tables[1])
