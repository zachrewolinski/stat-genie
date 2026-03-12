import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')

# Map columns to variables based on metadata and observed ranges
# deaths (skewed)
df['deaths'] = df['name']
# femininity ratings (1-11 scale from coders)
df['fem_rating'] = df['category']
# mturk rating (1-10-ish)
df['fem_rating_mturk'] = df['ind']
# binary female indicator
df['female_binary'] = df['masfem_mturk']
# severity controls
# wind speed at landfall (mph)
df['wind_speed'] = df['year']
# minimum pressure (mb)
df['min_pressure'] = df['ndam15']
# Saffir-Simpson category
df['gender_mf'] = df['gender_mf']
# year storm occurred
df['storm_year'] = df['wind']
# property damage (normalized)
df['damage_2013'] = df['elapsedyrs']

df['log_deaths'] = np.log1p(df['deaths'])
df['log_damage'] = np.log1p(df['damage_2013'])

results = {}

# Simple correlations
results['corr_fem_logdeaths'] = df['fem_rating'].corr(df['log_deaths'])
results['corr_fem_mturk_logdeaths'] = df['fem_rating_mturk'].corr(df['log_deaths'])
results['corr_female_binary_logdeaths'] = df['female_binary'].corr(df['log_deaths'])

# OLS with controls
formula = 'log_deaths ~ fem_rating + wind_speed + min_pressure + gender_mf + log_damage + storm_year'
model = smf.ols(formula, data=df).fit()

# Alternative with mturk rating
formula2 = 'log_deaths ~ fem_rating_mturk + wind_speed + min_pressure + gender_mf + log_damage + storm_year'
model2 = smf.ols(formula2, data=df).fit()

# Binary female indicator
formula3 = 'log_deaths ~ female_binary + wind_speed + min_pressure + gender_mf + log_damage + storm_year'
model3 = smf.ols(formula3, data=df).fit()

print('N:', len(df))
print('\nCorrelation fem_rating vs log_deaths:', results['corr_fem_logdeaths'])
print('Correlation fem_rating_mturk vs log_deaths:', results['corr_fem_mturk_logdeaths'])
print('Correlation female_binary vs log_deaths:', results['corr_female_binary_logdeaths'])

print('\nModel with fem_rating')
print(model.summary())
print('\nModel with fem_rating_mturk')
print(model2.summary())
print('\nModel with female_binary')
print(model3.summary())
