import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')
print('rows', df.shape)
print(df.head())

# Basic summary
print('\nMissing values per column:')
print(df.isna().sum())

# Create log deaths and log damage
# add 1 to handle zeros

df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam15'] = np.log1p(df['ndam15'])

# Standardize masfem for interpretability

df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std(ddof=0)

# severity controls: wind, min pressure, category
# Use OLS on log deaths

model1 = smf.ols('log_deaths ~ masfem_z + wind + min + category + log_ndam15 + year', data=df).fit()
print('\nOLS log_deaths ~ masfem_z + wind + min + category + log_ndam15 + year')
print(model1.summary())

# Poisson regression on deaths (count), with log link. add offset? not necessary; use robust SE.
# But we can use GLM Poisson with robust covariance.

model2 = smf.glm('alldeaths ~ masfem_z + wind + min + category + log_ndam15 + year',
                 data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson GLM alldeaths ~ masfem_z + wind + min + category + log_ndam15 + year')
print(model2.summary())

# Alternative: use gender_mf binary
model3 = smf.ols('log_deaths ~ gender_mf + wind + min + category + log_ndam15 + year', data=df).fit()
print('\nOLS log_deaths ~ gender_mf + wind + min + category + log_ndam15 + year')
print(model3.summary())

# Correlation
print('\nCorrelation masfem vs log_deaths:', df['masfem'].corr(df['log_deaths']))
print('Correlation gender_mf vs log_deaths:', df['gender_mf'].corr(df['log_deaths']))

# Model with interaction between masfem and intensity? use wind or category
model4 = smf.ols('log_deaths ~ masfem_z * wind + min + category + log_ndam15 + year', data=df).fit()
print('\nOLS log_deaths ~ masfem_z*wind + min + category + log_ndam15 + year')
print(model4.summary())

# Save key metrics
print('\nKey coefficients:')
print('model1 masfem_z coef', model1.params['masfem_z'], 'p', model1.pvalues['masfem_z'])
print('model2 masfem_z coef', model2.params['masfem_z'], 'p', model2.pvalues['masfem_z'])
print('model3 gender_mf coef', model3.params['gender_mf'], 'p', model3.pvalues['gender_mf'])
print('model4 masfem_z coef', model4.params['masfem_z'], 'p', model4.pvalues['masfem_z'])
print('model4 masfem_z:wind coef', model4.params['masfem_z:wind'], 'p', model4.pvalues['masfem_z:wind'])

