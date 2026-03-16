import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

path = 'hurricane.csv'
df = pd.read_csv(path)

# log1p deaths and damage

df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_dam'] = np.log1p(df['ndam'])

# report missing in key vars
key_vars = ['alldeaths','masfem','masfem_mturk','gender_mf','wind','min','category','ndam','year']
missing = df[key_vars].isna().sum()
print('missing counts')
print(missing)

# OLS with robust SE (HC3)
formula = 'log_deaths ~ masfem + wind + min + category + log_dam + year'
model = smf.ols(formula, data=df).fit(cov_type='HC3')
print('\nOLS (masfem) robust')
print(model.summary())

formula_mturk = 'log_deaths ~ masfem_mturk + wind + min + category + log_dam + year'
model_mturk = smf.ols(formula_mturk, data=df).fit(cov_type='HC3')
print('\nOLS (masfem_mturk) robust')
print(model_mturk.summary())

formula_gender = 'log_deaths ~ gender_mf + wind + min + category + log_dam + year'
model_gender = smf.ols(formula_gender, data=df).fit(cov_type='HC3')
print('\nOLS (gender_mf) robust')
print(model_gender.summary())

# Negative binomial GLM
formula_nb = 'alldeaths ~ masfem + wind + min + category + log_dam + year'
nb = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
print('\nNB (masfem)')
print(nb.summary())

formula_nb_mturk = 'alldeaths ~ masfem_mturk + wind + min + category + log_dam + year'
nb_mturk = smf.glm(formula_nb_mturk, data=df, family=sm.families.NegativeBinomial()).fit()
print('\nNB (masfem_mturk)')
print(nb_mturk.summary())

formula_nb_gender = 'alldeaths ~ gender_mf + wind + min + category + log_dam + year'
nb_gender = smf.glm(formula_nb_gender, data=df, family=sm.families.NegativeBinomial()).fit()
print('\nNB (gender_mf)')
print(nb_gender.summary())

# Simple comparisons by gender
print('\nMeans by gender_mf (0=male,1=female)')
print(df.groupby('gender_mf')['alldeaths'].mean())
print('\nMedian by gender_mf')
print(df.groupby('gender_mf')['alldeaths'].median())

# Simple correlation
print('\nCorr masfem vs log_deaths', df['masfem'].corr(df['log_deaths']))
print('Corr masfem_mturk vs log_deaths', df['masfem_mturk'].corr(df['log_deaths']))
