import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Create log fatalities (add 1 to handle zeros)
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Base covariates for storm intensity
covars = ['masfem', 'wind', 'min', 'category', 'year']

# Drop rows with missing in covariates or outcome
base_df = _df.dropna(subset=covars + ['log_deaths']).copy()

X = base_df[covars]
X = sm.add_constant(X)
y = base_df['log_deaths']

model = sm.OLS(y, X).fit()
print('Model 1: log_deaths ~ masfem + wind + min + category + year')
print(model.summary())

# Alternative model using MTurk femininity ratings
covars_mturk = ['masfem_mturk', 'wind', 'min', 'category', 'year']
base_df2 = _df.dropna(subset=covars_mturk + ['log_deaths']).copy()
X2 = sm.add_constant(base_df2[covars_mturk])
y2 = base_df2['log_deaths']
model2 = sm.OLS(y2, X2).fit()
print('\nModel 2: log_deaths ~ masfem_mturk + wind + min + category + year')
print(model2.summary())

# Simple correlation (no controls)
cor_masfem = _df['masfem'].corr(_df['log_deaths'])
cor_mturk = _df['masfem_mturk'].corr(_df['log_deaths'])
print('\nCorrelations with log_deaths:')
print(f"masfem corr: {cor_masfem:.3f}")
print(f"masfem_mturk corr: {cor_mturk:.3f}")
