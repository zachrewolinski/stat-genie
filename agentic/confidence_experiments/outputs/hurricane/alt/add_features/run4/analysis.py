import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
DF = pd.read_csv('hurricane.csv')

# Variables of interest
# Femininity measures
fem = DF['masfem']
sex = DF['gender_mf']

# Outcome: deaths (skewed) -> log1p
DF['log_deaths'] = np.log1p(DF['alldeaths'])

# Simple correlations
corr_masfem = stats.pearsonr(DF['masfem'], DF['log_deaths'])
corr_gender = stats.pearsonr(DF['gender_mf'], DF['log_deaths'])

# Regression with controls for storm intensity
# Use wind (higher = stronger), min pressure (lower = stronger), category
# include year to account for time trends
controls = ['wind', 'min', 'category', 'year']

X = DF[['masfem'] + controls]
X = sm.add_constant(X)
model = sm.OLS(DF['log_deaths'], X).fit()

X2 = DF[['gender_mf'] + controls]
X2 = sm.add_constant(X2)
model2 = sm.OLS(DF['log_deaths'], X2).fit()

# Also check model without year to see robustness
controls_no_year = ['wind', 'min', 'category']
X3 = DF[['masfem'] + controls_no_year]
X3 = sm.add_constant(X3)
model3 = sm.OLS(DF['log_deaths'], X3).fit()

# Output key results
print('n=', len(DF))
print('\nPearson correlations with log_deaths:')
print('masfem: r=%.3f p=%.4f' % (corr_masfem[0], corr_masfem[1]))
print('gender_mf: r=%.3f p=%.4f' % (corr_gender[0], corr_gender[1]))

print('\nOLS log_deaths ~ masfem + wind + min + category + year')
print(model.summary().tables[1])

print('\nOLS log_deaths ~ gender_mf + wind + min + category + year')
print(model2.summary().tables[1])

print('\nOLS log_deaths ~ masfem + wind + min + category')
print(model3.summary().tables[1])
