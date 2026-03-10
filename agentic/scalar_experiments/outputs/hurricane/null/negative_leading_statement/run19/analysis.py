import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic sanity
print('rows', len(df))

# Create log deaths
# deaths are counts, include zero; use log1p

df['log_deaths'] = np.log1p(df['alldeaths'])

# Key predictors
# Use masfem (1-11 rating) as main predictor

# Control variables: wind, min pressure, ndam (damage), category, year maybe.
# Include log damage due to skew

df['log_dam'] = np.log1p(df['ndam'])

# Fit OLS on log deaths
formula = 'log_deaths ~ masfem + wind + min + category + log_dam + year'
model = smf.ols(formula, data=df).fit()
print(model.summary())

# Also try Negative Binomial for count deaths
# Add constant and use GLM
formula_nb = 'alldeaths ~ masfem + wind + min + category + log_dam + year'
nb = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
print(nb.summary())

# Also check with gender_mf (binary) instead of masfem
formula2 = 'log_deaths ~ gender_mf + wind + min + category + log_dam + year'
model2 = smf.ols(formula2, data=df).fit()
print(model2.summary())

formula2_nb = 'alldeaths ~ gender_mf + wind + min + category + log_dam + year'
nb2 = smf.glm(formula2_nb, data=df, family=sm.families.NegativeBinomial()).fit()
print(nb2.summary())

# Correlations (simple)
print('corr masfem vs log_deaths', df['masfem'].corr(df['log_deaths']))
print('corr gender_mf vs log_deaths', df['gender_mf'].corr(df['log_deaths']))
