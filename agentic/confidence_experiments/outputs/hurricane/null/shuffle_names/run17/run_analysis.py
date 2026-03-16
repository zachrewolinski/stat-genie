import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('hurricane.csv')

# Map columns to meanings based on distributions
# name: deaths; category: femininity rating (1-11); ind: MTurk rating; masfem_mturk: binary gender
# gender_mf: storm category (1-5); year: wind speed; ndam15: minimum pressure; wind: year of storm

# Prepare variables

df = df.copy()

df['log_deaths'] = np.log1p(df['name'])

# Select model variables
predictors = ['category', 'gender_mf', 'year', 'ndam15', 'wind']
# drop missing
model_df = df[['log_deaths'] + predictors].dropna()

X = model_df[predictors]
X = sm.add_constant(X)

y = model_df['log_deaths']

model = sm.OLS(y, X).fit()
print(model.summary())

# Alternative with MTurk rating
predictors2 = ['ind', 'gender_mf', 'year', 'ndam15', 'wind']
model_df2 = df[['log_deaths'] + predictors2].dropna()
X2 = sm.add_constant(model_df2[predictors2])
model2 = sm.OLS(model_df2['log_deaths'], X2).fit()
print('\nMTurk rating model:')
print(model2.summary())

# Binary gender predictor
predictors3 = ['masfem_mturk', 'gender_mf', 'year', 'ndam15', 'wind']
model_df3 = df[['log_deaths'] + predictors3].dropna()
X3 = sm.add_constant(model_df3[predictors3])
model3 = sm.OLS(model_df3['log_deaths'], X3).fit()
print('\nBinary gender model:')
print(model3.summary())

# Simple correlation
corr = df[['category','ind','masfem_mturk','name','log_deaths']].corr()
print('\nCorrelation matrix (subset):')
print(corr)

