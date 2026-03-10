import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
_df = _df.copy()
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Variables
# Femininity: masfem (higher = more feminine). Also gender_mf (binary)

# Controls for severity and time
controls = ['wind', 'min', 'category', 'year']

# Drop rows with missing values in needed columns
cols = ['masfem', 'gender_mf', 'alldeaths', 'log_deaths'] + controls
_df = _df.dropna(subset=cols)

# Standardize some controls for interpretability (not required for p-values)
# We'll run models without standardization for simplicity

# Model 1: log deaths ~ masfem + controls
X1 = _df[['masfem'] + controls]
X1 = sm.add_constant(X1)
model1 = sm.OLS(_df['log_deaths'], X1).fit(cov_type='HC3')

# Model 2: log deaths ~ gender_mf + controls
X2 = _df[['gender_mf'] + controls]
X2 = sm.add_constant(X2)
model2 = sm.OLS(_df['log_deaths'], X2).fit(cov_type='HC3')

# Simple correlations
corr_masfem = _df[['masfem', 'log_deaths']].corr().iloc[0,1]

# Uncontrolled regression (for reference)
X0 = sm.add_constant(_df[['masfem']])
model0 = sm.OLS(_df['log_deaths'], X0).fit(cov_type='HC3')

print('N:', len(_df))
print('\nModel0: log_deaths ~ masfem')
print(model0.summary().tables[1])
print('\nModel1: log_deaths ~ masfem + controls')
print(model1.summary().tables[1])
print('\nModel2: log_deaths ~ gender_mf + controls')
print(model2.summary().tables[1])
print('\nCorrelation masfem-log_deaths:', corr_masfem)

# Additional model: include interactions or min? We'll keep simple.

