import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic checks
print('Rows:', len(_df))
print('Columns:', list(_df.columns))

# Focus variables
# Use log1p deaths to reduce skew, common in hurricane fatality modeling
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Simple correlation
corr = _df['masfem'].corr(_df['alldeaths'])
print('Correlation masfem vs alldeaths:', corr)

# OLS with controls for storm intensity and time
ols_vars = ['masfem', 'wind', 'min', 'category', 'year']
X = sm.add_constant(_df[ols_vars])
ols = sm.OLS(_df['log_deaths'], X).fit()
print('\nOLS log_deaths ~ masfem + controls')
print(ols.summary())

# Alternative masculinity rating (mturk)
ols_vars_mturk = ['masfem_mturk', 'wind', 'min', 'category', 'year']
X2 = sm.add_constant(_df[ols_vars_mturk])
ols2 = sm.OLS(_df['log_deaths'], X2).fit()
print('\nOLS log_deaths ~ masfem_mturk + controls')
print(ols2.summary())

# Binary female name indicator
ols_vars_gender = ['gender_mf', 'wind', 'min', 'category', 'year']
X3 = sm.add_constant(_df[ols_vars_gender])
ols3 = sm.OLS(_df['log_deaths'], X3).fit()
print('\nOLS log_deaths ~ gender_mf + controls')
print(ols3.summary())

# Negative binomial for count-like deaths
# Add a small offset? Not needed for counts; just use NB with same controls
nb = sm.GLM(
    _df['alldeaths'],
    sm.add_constant(_df[ols_vars]),
    family=sm.families.NegativeBinomial()
).fit()
print('\nNegative Binomial alldeaths ~ masfem + controls')
print(nb.summary())

# Collect key coefficients
results = {
    'corr_masfem_deaths': corr,
    'ols_masfem_coef': ols.params['masfem'],
    'ols_masfem_p': ols.pvalues['masfem'],
    'ols_mturk_coef': ols2.params['masfem_mturk'],
    'ols_mturk_p': ols2.pvalues['masfem_mturk'],
    'ols_gender_coef': ols3.params['gender_mf'],
    'ols_gender_p': ols3.pvalues['gender_mf'],
    'nb_masfem_coef': nb.params['masfem'],
    'nb_masfem_p': nb.pvalues['masfem'],
}
print('\nKey results:', results)
