import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns to clearer names based on value ranges and provided metadata
renamed = df.rename(columns={
    'ndam': 'id',                 # unique row id
    'wind': 'year',               # year of hurricane
    'alldeaths': 'storm_name',    # hurricane name
    'category': 'feminity_index', # 1-11 femininity rating
    'ndam15': 'min_pressure',     # minimum pressure at landfall
    'masfem_mturk': 'female_binary',
    'gender_mf': 'saffir_category',
    'name': 'deaths',             # total deaths
    'elapsedyrs': 'damage_2013',  # normalized damage (2013 dollars)
    'masfem': 'elapsed_years',    # years since hurricane
    'min': 'source',
    'ind': 'mturk_feminity',
    'year': 'wind_mph',           # max wind speed
    'source': 'damage_2015'       # normalized damage (2015 dollars)
})

# Basic sanity checks
print('Rows:', renamed.shape[0])
print('Death summary:\n', renamed['deaths'].describe())
print('Femininity index summary:\n', renamed['feminity_index'].describe())
print('Binary female counts:\n', renamed['female_binary'].value_counts())

# Simple correlation
corr = renamed['feminity_index'].corr(renamed['deaths'])
print('Correlation (feminity_index vs deaths):', corr)

# Regression: log(deaths + 1) on femininity, controlling for intensity
renamed['log_deaths'] = np.log1p(renamed['deaths'])

X = renamed[['feminity_index', 'wind_mph', 'min_pressure', 'saffir_category', 'damage_2015']]
X = sm.add_constant(X)
model = sm.OLS(renamed['log_deaths'], X).fit()
print('\nOLS model with feminity_index:')
print(model.summary())

# Alternative specifications with different femininity measures
X2 = renamed[['mturk_feminity', 'wind_mph', 'min_pressure', 'saffir_category', 'damage_2015']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(renamed['log_deaths'], X2).fit()
print('\nOLS model with mturk_feminity:')
print(model2.summary())

X3 = renamed[['female_binary', 'wind_mph', 'min_pressure', 'saffir_category', 'damage_2015']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(renamed['log_deaths'], X3).fit()
print('\nOLS model with female_binary:')
print(model3.summary())

