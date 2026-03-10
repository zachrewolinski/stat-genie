import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
raw = pd.read_csv('hurricane.csv')

# Infer semantic columns based on metadata and value ranges
# id
# year of hurricane
# name (string)
# femininity rating (continuous, 1-11)
# min pressure (mb)
# female indicator (0/1)
# Saffir-Simpson category (1-5)
# deaths
# damage normalized (2013)
# elapsed years since hurricane
# source
# MTurk femininity rating (continuous)
# max wind speed (mph)
# damage normalized (2015)

_df = raw.copy()
_df = _df.rename(columns={
    'ndam': 'id',
    'wind': 'year',
    'alldeaths': 'name',
    'category': 'femininity_coders',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_indicator',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'elapsed_years',
    'min': 'source',
    'ind': 'femininity_mturk',
    'year': 'max_wind',
    'source': 'damage_2015',
})

# Basic check
print(_df[['year','name','femininity_coders','femininity_mturk','female_indicator','ss_category','max_wind','min_pressure','deaths','damage_2013','damage_2015']].head())

# Outcome: deaths (skewed) -> log1p
_df['log_deaths'] = np.log1p(_df['deaths'])

# Controls: storm intensity proxies
controls = ['max_wind','min_pressure','ss_category']

# Model 1: femininity (coder rating)
X1 = sm.add_constant(_df[['femininity_coders'] + controls])
model1 = sm.OLS(_df['log_deaths'], X1, missing='drop').fit(cov_type='HC3')

# Model 2: femininity MTurk rating
X2 = sm.add_constant(_df[['femininity_mturk'] + controls])
model2 = sm.OLS(_df['log_deaths'], X2, missing='drop').fit(cov_type='HC3')

# Model 3: binary female indicator
X3 = sm.add_constant(_df[['female_indicator'] + controls])
model3 = sm.OLS(_df['log_deaths'], X3, missing='drop').fit(cov_type='HC3')

# Print summaries
print('\nModel 1 (coder femininity):')
print(model1.summary().tables[1])
print('\nModel 2 (mturk femininity):')
print(model2.summary().tables[1])
print('\nModel 3 (female indicator):')
print(model3.summary().tables[1])

# Simple correlations (Spearman) between femininity and deaths
corrs = {}
for col in ['femininity_coders','femininity_mturk','female_indicator']:
    corrs[col] = _df[['deaths', col]].corr(method='spearman').iloc[0,1]
print('\nSpearman correlations with deaths:', corrs)

# Check if femininity predicts more deaths without controls
X1_simple = sm.add_constant(_df[['femininity_coders']])
model1_simple = sm.OLS(_df['log_deaths'], X1_simple, missing='drop').fit(cov_type='HC3')
print('\nModel 1 simple:')
print(model1_simple.summary().tables[1])
