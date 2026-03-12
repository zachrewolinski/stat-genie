import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Column mapping from info.json descriptions
# category: femininity index (1-11)
# masfem_mturk: binary gender indicator
# ind: MTurk rating scale (1-11)
# name: total deaths
# ndam15: minimum pressure at landfall
# year: max wind speed at landfall
# gender_mf: Saffir-Simpson category
# wind: year of hurricane

# Prepare variables
# Ensure numeric
num_cols = ['category','masfem_mturk','ind','name','ndam15','year','gender_mf','wind','elapsedyrs','source','masfem']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Outcome: deaths
# Use log1p to handle skew

df['log_deaths'] = np.log1p(df['name'])

# Femininity measures
fem_vars = ['category','ind','masfem_mturk']

results = {}

# Simple correlations
for fem in fem_vars:
    if fem in df.columns:
        corr = df[['name', fem]].corr().iloc[0,1]
        corr_log = df[['log_deaths', fem]].corr().iloc[0,1]
        results[f'corr_{fem}_deaths'] = corr
        results[f'corr_{fem}_logdeaths'] = corr_log

# Regression controls for storm severity and year
controls = ['gender_mf','year','ndam15','wind']

# Build regression for each fem var
reg_results = {}
for fem in fem_vars:
    cols = [fem] + controls
    sub = df[cols + ['log_deaths']].dropna()
    X = sub[cols]
    X = sm.add_constant(X)
    y = sub['log_deaths']
    model = sm.OLS(y, X).fit(cov_type='HC3')
    reg_results[fem] = {
        'n': int(model.nobs),
        'coef': float(model.params[fem]),
        'se': float(model.bse[fem]),
        'p': float(model.pvalues[fem]),
        'ci_low': float(model.conf_int().loc[fem,0]),
        'ci_high': float(model.conf_int().loc[fem,1])
    }

# Also run with raw deaths (overdispersed) but still OLS for sense
reg_results_raw = {}
for fem in fem_vars:
    cols = [fem] + controls
    sub = df[cols + ['name']].dropna()
    X = sm.add_constant(sub[cols])
    y = sub['name']
    model = sm.OLS(y, X).fit(cov_type='HC3')
    reg_results_raw[fem] = {
        'n': int(model.nobs),
        'coef': float(model.params[fem]),
        'se': float(model.bse[fem]),
        'p': float(model.pvalues[fem]),
        'ci_low': float(model.conf_int().loc[fem,0]),
        'ci_high': float(model.conf_int().loc[fem,1])
    }

# Summaries
print('Correlation results:')
for k,v in results.items():
    print(k, v)

print('\nLog-deaths regression (HC3):')
for fem, res in reg_results.items():
    print(fem, res)

print('\nRaw deaths regression (HC3):')
for fem, res in reg_results_raw.items():
    print(fem, res)
