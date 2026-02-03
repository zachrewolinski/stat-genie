import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('hurricane.csv')

# Basic prep

df['log_deaths'] = np.log1p(df['alldeaths'])

# Core controls: intensity metrics available at landfall
controls = ['category', 'wind', 'min']

# Model 1: femininity only
X1 = sm.add_constant(df[['masfem']])
model1 = sm.OLS(df['log_deaths'], X1).fit(cov_type='HC3')

# Model 2: femininity + controls
X2 = sm.add_constant(df[['masfem'] + controls])
model2 = sm.OLS(df['log_deaths'], X2).fit(cov_type='HC3')

# Model 3: femininity + controls + gender indicator (redundant but a check)
X3 = sm.add_constant(df[['masfem', 'gender_mf'] + controls])
model3 = sm.OLS(df['log_deaths'], X3).fit(cov_type='HC3')

# Simple correlation
corr = df['masfem'].corr(df['log_deaths'])

print('N =', len(df))
print('Correlation masfem vs log_deaths:', corr)
print('\nModel 1: log_deaths ~ masfem')
print(model1.summary())
print('\nModel 2: log_deaths ~ masfem + category + wind + min')
print(model2.summary())
print('\nModel 3: log_deaths ~ masfem + gender_mf + category + wind + min')
print(model3.summary())

# Key coefficient table
for i, m in enumerate([model1, model2, model3], start=1):
    coef = m.params.get('masfem', np.nan)
    pval = m.pvalues.get('masfem', np.nan)
    print(f'Model {i} masfem coef={coef:.4f}, p={pval:.4f}')
