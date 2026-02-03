import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('hurricane.csv')

# Outcome: log(1 + deaths) to reduce skew
df['log_deaths'] = np.log1p(df['alldeaths'])

# Primary predictor: femininity score (masfem)
# Controls: wind speed, minimum pressure, category, inflation-adjusted damage, year
controls = ['masfem', 'wind', 'min', 'category', 'ndam15', 'year']
X = sm.add_constant(df[controls])
model = sm.OLS(df['log_deaths'], X, missing='drop').fit()
robust = model.get_robustcov_results(cov_type='HC3')

print('Rows:', len(df))
print('Correlation masfem vs log_deaths:', df['masfem'].corr(df['log_deaths']))
print('\nOLS (HC3 robust SE) summary:')
print(robust.summary())

# Alternative binary gender indicator
controls_bin = ['gender_mf', 'wind', 'min', 'category', 'ndam15', 'year']
X2 = sm.add_constant(df[controls_bin])
model2 = sm.OLS(df['log_deaths'], X2, missing='drop').fit()
robust2 = model2.get_robustcov_results(cov_type='HC3')

print('\nOLS with gender_mf (HC3 robust SE) summary:')
print(robust2.summary())
