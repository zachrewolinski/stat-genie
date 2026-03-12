import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns by observed ranges
# Actual year is in column 'wind'
# Hurricane name in 'alldeaths'
# Femininity rating (1-11) in 'category'
# Min pressure in 'ndam15'
# Binary gender in 'masfem_mturk'
# Saffir-Simpson category in 'gender_mf'
# Deaths in 'name'
# Damage (2013 adj?) in 'elapsedyrs'
# Damage (2015 adj?) in 'source'
# MTurk femininity rating in 'ind'
# Max wind speed in 'year'

df = df.copy()

# Prepare variables

df['deaths'] = df['name']
df['log_deaths'] = np.log1p(df['deaths'])

df['fem_score'] = df['category']
df['fem_mturk'] = df['ind']
df['fem_binary'] = df['masfem_mturk']

df['wind_speed'] = df['year']
df['min_pressure'] = df['ndam15']
df['ss_category'] = df['gender_mf']

df['storm_year'] = df['wind']
df['damage_2013'] = df['elapsedyrs']
df['damage_2015'] = df['source']

# Basic comparisons
mean_deaths_by_gender = df.groupby('fem_binary')['deaths'].agg(['mean','median','count'])

# Correlations
corr_fem_deaths = df[['fem_score','deaths']].corr().iloc[0,1]

# OLS models

def fit_ols(y, X):
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()

m1 = fit_ols(df['log_deaths'], df[['fem_score']])
m2 = fit_ols(df['log_deaths'], df[['fem_score','wind_speed','min_pressure','ss_category','storm_year']])
m3 = fit_ols(df['log_deaths'], df[['fem_score','wind_speed','min_pressure','ss_category','storm_year','damage_2015']])

# Also use MTurk rating
m4 = fit_ols(df['log_deaths'], df[['fem_mturk','wind_speed','min_pressure','ss_category','storm_year']])

# Extract key stats
results = {
    'mean_deaths_by_gender': mean_deaths_by_gender,
    'corr_fem_deaths': corr_fem_deaths,
    'm1': m1,
    'm2': m2,
    'm3': m3,
    'm4': m4,
}

print('Mean deaths by gender (0=male,1=female):')
print(mean_deaths_by_gender)
print('\nCorrelation fem_score vs deaths:', corr_fem_deaths)

for name, model in [('m1',m1),('m2',m2),('m3',m3),('m4',m4)]:
    print('\n', name)
    print(model.summary().tables[1])

