import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.width', 200)

path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns to meaningful names based on ranges
# wind -> year of storm, year -> max wind speed
# category -> femininity index (1-11), ind -> mturk femininity index
# name -> deaths

# create variables

df = df.copy()

df['deaths'] = df['name']
df['log_deaths'] = np.log1p(df['deaths'])

df['fem_index'] = df['category']
df['fem_mturk'] = df['ind']
df['female_binary'] = df['masfem_mturk']

df['storm_year'] = df['wind']
df['wind_speed'] = df['year']
df['min_pressure'] = df['ndam15']
df['ss_category'] = df['gender_mf']

# OLS with log deaths
formula = 'log_deaths ~ fem_index + wind_speed + min_pressure + ss_category + storm_year'
model = smf.ols(formula, data=df).fit()
print(model.summary())

formula2 = 'log_deaths ~ fem_mturk + wind_speed + min_pressure + ss_category + storm_year'
model2 = smf.ols(formula2, data=df).fit()
print(model2.summary())

# binary female indicator
formula3 = 'log_deaths ~ female_binary + wind_speed + min_pressure + ss_category + storm_year'
model3 = smf.ols(formula3, data=df).fit()
print(model3.summary())

# Poisson GLM (deaths counts) for robustness
poisson1 = smf.glm('deaths ~ fem_index + wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.Poisson()).fit()
print(poisson1.summary())

poisson2 = smf.glm('deaths ~ fem_mturk + wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.Poisson()).fit()
print(poisson2.summary())

poisson3 = smf.glm('deaths ~ female_binary + wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.Poisson()).fit()
print(poisson3.summary())

