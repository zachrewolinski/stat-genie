import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('hurricane.csv')

# Core variables
# Use log1p to handle zero deaths/damages

df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam15'] = np.log1p(df['ndam15'])

# Model: deaths vs femininity controlling for storm intensity and exposure proxies
# Controls chosen from dataset description: wind speed, minimum pressure, category, year, damages

X = df[['masfem', 'wind', 'min', 'category', 'year', 'log_ndam15']].copy()
X = sm.add_constant(X)

y = df['log_deaths']

model = sm.OLS(y, X).fit()

print(model.summary())

# Simple decision rule
coef = model.params['masfem']
pval = model.pvalues['masfem']
print('\nMasfem coef:', coef)
print('Masfem p-value:', pval)

# Also show a simpler model with fewer controls

X2 = df[['masfem', 'wind', 'min', 'category']].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2).fit()
print('\nSimpler model summary:')
print(model2.summary())
print('\nMasfem coef (simple):', model2.params['masfem'])
print('Masfem p-value (simple):', model2.pvalues['masfem'])
