import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns to inferred meanings based on values
# wind column is year (1950-2012)
# alldeaths is hurricane name (string)
# category is femininity rating (1-11 scale)
# ndam15 is minimum pressure (hPa)
# masfem_mturk is binary female indicator
# gender_mf is Saffir-Simpson category (1-5)
# name is deaths (count)
# elapsedyrs is normalized damage (2013$)
# masfem is years elapsed since hurricane
# min is data source
# ind is MTurk femininity rating (1-11 scale)
# year is wind speed (mph)
# source is normalized damage (2015$)

df = _df.copy()

# Rename for clarity
rename_map = {
    'wind': 'year',
    'alldeaths': 'storm_name',
    'category': 'fem_rating',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'storm_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'elapsed_years',
    'min': 'data_source',
    'ind': 'mturk_fem_rating',
    'year': 'wind_speed',
    'source': 'damage_2015',
}

df = df.rename(columns=rename_map)

# Derived variables
_df_valid = df.copy()
_df_valid['log_deaths'] = np.log1p(_df_valid['deaths'])

# Define controls
controls = ['wind_speed', 'min_pressure', 'storm_category', 'year']

# Prepare data for modeling, drop rows with missing values
model_cols_fem = ['log_deaths', 'fem_rating'] + controls
model_cols_mturk = ['log_deaths', 'mturk_fem_rating'] + controls

fem_data = _df_valid[model_cols_fem].dropna()
mturk_data = _df_valid[model_cols_mturk].dropna()

# Helper to run OLS with robust SEs

def robust_ols(y, X):
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model

# Simple bivariate model
simple_fem = robust_ols(_df_valid['log_deaths'], _df_valid[['fem_rating']])

# Controlled models
ctrl_fem = robust_ols(fem_data['log_deaths'], fem_data[['fem_rating'] + controls])
ctrl_mturk = robust_ols(mturk_data['log_deaths'], mturk_data[['mturk_fem_rating'] + controls])

# Spearman correlations
spearman_fem = _df_valid[['deaths', 'fem_rating']].corr(method='spearman').iloc[0,1]
spearman_mturk = _df_valid[['deaths', 'mturk_fem_rating']].corr(method='spearman').iloc[0,1]

# Summaries
print('N total:', len(_df_valid))
print('N fem model:', len(fem_data))
print('N mturk model:', len(mturk_data))

print('\nSpearman correlations:')
print('deaths vs fem_rating:', spearman_fem)
print('deaths vs mturk_fem_rating:', spearman_mturk)

print('\nSimple model (log_deaths ~ fem_rating):')
print(simple_fem.summary().tables[1])

print('\nControlled model (log_deaths ~ fem_rating + controls):')
print(ctrl_fem.summary().tables[1])

print('\nControlled model (log_deaths ~ mturk_fem_rating + controls):')
print(ctrl_mturk.summary().tables[1])
