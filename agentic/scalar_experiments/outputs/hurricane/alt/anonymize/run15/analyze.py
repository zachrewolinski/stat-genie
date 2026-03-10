import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.columns)

# Rename columns for readability
cols = {c: c for c in df.columns}
# We'll map to meaningful names based on info.json
mapping = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem',
    'feature5': 'min_pressure',
    'feature6': 'female',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'masfem_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=mapping)

# Basic correlations between masfem/female and deaths
print('corr masfem deaths', df['masfem'].corr(df['deaths']))
print('corr female deaths', df['female'].corr(df['deaths']))

# log-transform deaths and damage to reduce skew
# add 1 to deaths as may be zeros

df['log_deaths'] = np.log1p(df['deaths'])
df['log_damage_2013'] = np.log1p(df['damage_2013'])
df['log_damage_2015'] = np.log1p(df['damage_2015'])

# Regression: log_deaths ~ masfem + controls (min_pressure, category, max_wind, year)
# Use robust SE (HC3)
formula = 'log_deaths ~ masfem + min_pressure + category + max_wind + year'
model = smf.ols(formula, data=df).fit(cov_type='HC3')
print(model.summary())

formula2 = 'log_deaths ~ female + min_pressure + category + max_wind + year'
model2 = smf.ols(formula2, data=df).fit(cov_type='HC3')
print(model2.summary())

# Alternative with damage control
formula3 = 'log_deaths ~ masfem + min_pressure + category + max_wind + log_damage_2013 + year'
model3 = smf.ols(formula3, data=df).fit(cov_type='HC3')
print(model3.summary())

# Simple bivariate
model4 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
print(model4.summary())

