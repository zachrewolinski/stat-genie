import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')
print(_df.head())
print(_df.describe(include='all'))

# Basic sanity
print('rows', len(_df))
print('missing', _df.isna().sum())

# create log deaths
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# simple correlation
print('corr masfem vs deaths', _df['masfem'].corr(_df['alldeaths']))
print('corr masfem vs log deaths', _df['masfem'].corr(_df['log_deaths']))

# simple regression
m1 = smf.ols('log_deaths ~ masfem', data=_df).fit()
print(m1.summary())

# controls for severity
m2 = smf.ols('log_deaths ~ masfem + wind + min + category + ndam15 + year', data=_df).fit()
print(m2.summary())

# interaction with severity (wind)
m3 = smf.ols('log_deaths ~ masfem * wind + min + category + ndam15 + year', data=_df).fit()
print(m3.summary())

# alt gender_mf binary
m4 = smf.ols('log_deaths ~ gender_mf + wind + min + category + ndam15 + year', data=_df).fit()
print(m4.summary())

# interaction gender with wind
m5 = smf.ols('log_deaths ~ gender_mf * wind + min + category + ndam15 + year', data=_df).fit()
print(m5.summary())
