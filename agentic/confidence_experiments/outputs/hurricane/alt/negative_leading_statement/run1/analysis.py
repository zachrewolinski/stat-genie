import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Prepare variables
# Log-transform deaths to handle skew; add 1 to include zeros
_df['log_deaths'] = np.log(_df['alldeaths'] + 1)

# Basic correlations
corr_masfem = _df['masfem'].corr(_df['log_deaths'])
corr_gender = _df['gender_mf'].corr(_df['log_deaths'])

# Regression: log_deaths ~ masfem + wind + min + category + year
# Add intercept
X = _df[['masfem', 'wind', 'min', 'category', 'year']].copy()
X = sm.add_constant(X)
y = _df['log_deaths']
model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

# Alternative with masfem_mturk instead of masfem
X2 = _df[['masfem_mturk', 'wind', 'min', 'category', 'year']].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2, missing='drop').fit(cov_type='HC3')

# Binary gender model
X3 = _df[['gender_mf', 'wind', 'min', 'category', 'year']].copy()
X3 = sm.add_constant(X3)
model3 = sm.OLS(y, X3, missing='drop').fit(cov_type='HC3')

print('n=', len(_df))
print('corr masfem vs log_deaths:', corr_masfem)
print('corr gender_mf vs log_deaths:', corr_gender)
print('\nOLS log_deaths ~ masfem + wind + min + category + year (HC3)')
print(model.summary().tables[1])
print('\nOLS log_deaths ~ masfem_mturk + wind + min + category + year (HC3)')
print(model2.summary().tables[1])
print('\nOLS log_deaths ~ gender_mf + wind + min + category + year (HC3)')
print(model3.summary().tables[1])
