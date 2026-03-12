import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.max_columns', None)

df = pd.read_csv('hurricane.csv')

# Define variables based on observed ranges
# Outcome: deaths
# Femininity measures: category (1-11 scale), ind (1-11 scale), masfem_mturk (binary)
# Controls: wind speed (year col), min pressure (ndam15), Saffir category (gender_mf), damage (source), year (wind)

df = df.copy()

df['log_deaths'] = np.log1p(df['name'])
df['log_damage'] = np.log1p(df['source'])

controls = ['year', 'ndam15', 'gender_mf', 'log_damage', 'wind']

results = {}

for fem in ['category', 'ind', 'masfem_mturk']:
    cols = ['log_deaths', fem] + controls
    d = df[cols].dropna().copy()
    formula = 'log_deaths ~ ' + fem + ' + ' + ' + '.join(controls)
    model = smf.ols(formula, data=d).fit(cov_type='HC3')
    results[fem] = model

print('N rows:', len(df))
print('\nCorrelation (femininity vs deaths):')
for fem in ['category','ind','masfem_mturk']:
    print(fem, df['name'].corr(df[fem]))

print('\nOLS results (log deaths) with controls:')
for fem, model in results.items():
    print('\n', fem)
    print(model.summary().tables[1])

# Simple group comparison for binary gender
if df['masfem_mturk'].nunique()==2:
    grp = df.groupby('masfem_mturk')['name'].agg(['mean','median','count'])
    print('\nDeaths by masfem_mturk group:')
    print(grp)

