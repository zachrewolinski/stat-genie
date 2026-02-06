import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Outcome: log deaths to reduce skew
_df['log_deaths'] = np.log(_df['alldeaths'])

# Simple model: femininity score only
X1 = sm.add_constant(_df[['masfem']])
model1 = sm.OLS(_df['log_deaths'], X1).fit()

# Controlled model: account for storm severity and time
controls = ['wind', 'min', 'category', 'ndam15', 'year']
X2 = sm.add_constant(_df[['masfem'] + controls])
model2 = sm.OLS(_df['log_deaths'], X2).fit()

# Alternative gender indicator
X3 = sm.add_constant(_df[['gender_mf'] + controls])
model3 = sm.OLS(_df['log_deaths'], X3).fit()

# Print key results
print('Model 1: log_deaths ~ masfem')
print(model1.summary())
print('\nModel 2: log_deaths ~ masfem + controls')
print(model2.summary())
print('\nModel 3: log_deaths ~ gender_mf + controls')
print(model3.summary())
