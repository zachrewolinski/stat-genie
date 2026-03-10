import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns to semantic names based on inspection
# (column names are shuffled)
df = _df.rename(columns={
    'alldeaths': 'name',   # hurricane name
    'name': 'deaths',      # total deaths
    'wind': 'year',        # year of hurricane
    'year': 'wind',        # max wind speed
    'gender_mf': 'category',  # Saffir-Simpson category
    'category': 'masfem',  # femininity rating (coder-based)
    'ind': 'masfem_mturk', # MTurk femininity rating
    'masfem_mturk': 'female', # binary female name
    'ndam15': 'min_pressure',
    'elapsedyrs': 'damage_2013',
    'source': 'damage_2015',
    'masfem': 'elapsed_years',
    'min': 'source_type',
    'ndam': 'id'
})

# Basic sanity checks
print('Columns after rename:', df.columns.tolist())

# Prepare variables
# Use log(deaths+1) due to skew
# Use continuous femininity scores (masfem and masfem_mturk) and binary female indicator
for col in ['deaths', 'masfem', 'masfem_mturk', 'female', 'wind', 'min_pressure', 'category', 'year']:
    if col not in df.columns:
        raise ValueError(f'missing {col}')

# Build datasets for regression
reg_df = df[['deaths','masfem','masfem_mturk','female','wind','min_pressure','category','year']].copy()
reg_df['log_deaths'] = np.log1p(reg_df['deaths'])

# Drop rows with missing values in any predictor
reg_df = reg_df.dropna()

print('N used:', len(reg_df))

# Model 1: log deaths ~ masfem + severity controls
model1 = smf.ols('log_deaths ~ masfem + wind + min_pressure + category + year', data=reg_df).fit()
print('\nModel 1 (masfem):')
print(model1.summary())

# Model 2: log deaths ~ masfem_mturk + controls
model2 = smf.ols('log_deaths ~ masfem_mturk + wind + min_pressure + category + year', data=reg_df).fit()
print('\nModel 2 (masfem_mturk):')
print(model2.summary())

# Model 3: log deaths ~ female + controls
model3 = smf.ols('log_deaths ~ female + wind + min_pressure + category + year', data=reg_df).fit()
print('\nModel 3 (female binary):')
print(model3.summary())

# Simple bivariate correlations
print('\nCorrelations with log_deaths:')
print(reg_df[['log_deaths','masfem','masfem_mturk','female','wind','min_pressure','category','year']].corr()['log_deaths'])

# Optional interaction with severity (wind) like in original paper
model4 = smf.ols('log_deaths ~ masfem * wind + min_pressure + category + year', data=reg_df).fit()
print('\nModel 4 (masfem * wind):')
print(model4.summary())

