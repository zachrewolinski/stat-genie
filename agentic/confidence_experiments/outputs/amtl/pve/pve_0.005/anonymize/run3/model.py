import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Derived rate (can be negative due to anonymization)

df['rate'] = df['feature3'] / df['feature4']

# indicator for Homo sapiens

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# WLS on rate with weights = observable sockets (feature4)

# Use formula for categorical tooth class
formula = 'rate ~ is_human + feature5 + feature7 + C(feature1)'

model_wls = smf.wls(formula, data=df, weights=df['feature4']).fit(cov_type='HC3')

print('WLS rate model (HC3)')
print(model_wls.summary())

# OLS on rate with robust SE
model_ols = smf.ols(formula, data=df).fit(cov_type='HC3')
print('\nOLS rate model (HC3)')
print(model_ols.summary())

# Model with genus categories
formula_genus = 'rate ~ C(feature8) + feature5 + feature7 + C(feature1)'
model_genus = smf.wls(formula_genus, data=df, weights=df['feature4']).fit(cov_type='HC3')
print('\nWLS rate model with genus (HC3)')
print(model_genus.summary())

# effect of Homo sapiens vs others: coefficient in is_human
print('\nIs_human coef (WLS):', model_wls.params['is_human'], 'p', model_wls.pvalues['is_human'])

# Compare mean rates by genus (weighted)
weighted_mean = df.groupby('feature8').apply(lambda g: np.average(g['rate'], weights=g['feature4']))
print('\nWeighted mean rate by genus:')
print(weighted_mean)

