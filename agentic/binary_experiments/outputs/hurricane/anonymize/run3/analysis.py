import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset

df = pd.read_csv('hurricane.csv')

# Core variables
# feature4: femininity index (1-11)
# feature6: binary gender indicator (0=male,1=female)
# feature8: deaths
# feature7: category
# feature5: min pressure
# feature13: max wind speed
# feature2: year
# feature14: normalized damage

# Prepare analysis dataset
analysis_cols = ['feature8','feature4','feature6','feature7','feature5','feature13','feature2','feature14']
d = df[analysis_cols].copy()

# Transformations
# Use log1p for deaths and damages to reduce skew

d['log_deaths'] = np.log1p(d['feature8'])
d['log_damage'] = np.log1p(d['feature14'])

# Bivariate correlations
corr_fem = d[['feature4','log_deaths']].corr().iloc[0,1]
corr_bin = d[['feature6','log_deaths']].corr().iloc[0,1]

print('Bivariate correlations with log(deaths):')
print(f'  Femininity index vs log(deaths): {corr_fem:.3f}')
print(f'  Binary female vs log(deaths): {corr_bin:.3f}')

# Regression with femininity index
X_fem = d[['feature4','feature7','feature5','feature13','feature2','log_damage']]
X_fem = sm.add_constant(X_fem)
model_fem = sm.OLS(d['log_deaths'], X_fem, missing='drop').fit()

print('\nOLS on log(deaths) with femininity index + controls:')
print(model_fem.summary())

# Regression with binary gender
X_bin = d[['feature6','feature7','feature5','feature13','feature2','log_damage']]
X_bin = sm.add_constant(X_bin)
model_bin = sm.OLS(d['log_deaths'], X_bin, missing='drop').fit()

print('\nOLS on log(deaths) with binary gender + controls:')
print(model_bin.summary())

# Extract key coefficients
coef_fem = model_fem.params['feature4']
p_fem = model_fem.pvalues['feature4']
coef_bin = model_bin.params['feature6']
p_bin = model_bin.pvalues['feature6']

print('\nKey coefficients:')
print(f'  Femininity index coefficient: {coef_fem:.4f}, p-value: {p_fem:.3f}')
print(f'  Binary female coefficient: {coef_bin:.4f}, p-value: {p_bin:.3f}')
