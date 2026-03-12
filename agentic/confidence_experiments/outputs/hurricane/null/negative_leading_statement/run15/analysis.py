import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = 'hurricane.csv'

df = pd.read_csv(DATA_PATH)

# Basic variables
# Outcome: fatalities (alldeaths). Use log1p due to skew.
# Predictor: name femininity (masfem) and binary gender (gender_mf).
# Controls: storm intensity (wind, min pressure), category, damage (ndam15), year.

# Prepare data
cols = ['alldeaths','masfem','gender_mf','wind','min','category','ndam15','year']

# Remove rows with missing values
sub = df[cols].dropna().copy()

sub['log_deaths'] = np.log1p(sub['alldeaths'])
sub['log_ndam15'] = np.log1p(sub['ndam15'])

# Simple correlation between masfem and alldeaths
corr = sub['masfem'].corr(sub['alldeaths'])

# OLS regression: log deaths ~ masfem + controls
X = sub[['masfem','wind','min','category','log_ndam15','year']]
X = sm.add_constant(X)
model = sm.OLS(sub['log_deaths'], X).fit(cov_type='HC3')

# Alternative model using binary gender
X2 = sub[['gender_mf','wind','min','category','log_ndam15','year']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(sub['log_deaths'], X2).fit(cov_type='HC3')

print('n', len(sub))
print('corr_masfem_alldeaths', corr)
print('model_masfem_summary')
print(model.summary())
print('model_gender_summary')
print(model2.summary())

# Extract key stats
coef = model.params['masfem']
pval = model.pvalues['masfem']
coef2 = model2.params['gender_mf']
pval2 = model2.pvalues['gender_mf']

print('masfem_coef', coef, 'p', pval)
print('gender_coef', coef2, 'p', pval2)

# Also check bivariate OLS (no controls)
Xb = sm.add_constant(sub[['masfem']])
model_b = sm.OLS(sub['log_deaths'], Xb).fit(cov_type='HC3')
print('bivariate_masfem')
print(model_b.summary())
print('bivariate_masfem_coef', model_b.params['masfem'], 'p', model_b.pvalues['masfem'])

