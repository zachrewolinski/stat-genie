import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns to inferred meanings
# (Column names are shuffled in this dataset.)
col_map = {
    'storm_id': 'ndam',
    'year': 'wind',
    'storm_name': 'alldeaths',
    'femininity_rating': 'category',  # 1-11 scale from coders
    'min_pressure': 'ndam15',         # mb
    'fem_binary': 'masfem_mturk',      # 0=male, 1=female
    'category_ss': 'gender_mf',        # Saffir-Simpson category 1-5
    'deaths': 'name',                  # total deaths
    'damage_norm_2013': 'elapsedyrs',  # normalized damage (approx)
    'elapsed_years': 'masfem',         # years since hurricane (1-63)
    'data_source': 'min',
    'femininity_mturk': 'ind',         # MTurk average rating
    'wind_mph': 'year',                # max wind speed
    'damage_norm_2015': 'source',      # normalized damage (approx)
}

# Build analysis dataframe
_df2 = _df.rename(columns={v: k for k, v in col_map.items()})

# Keep relevant columns and drop rows with missing values
cols = [
    'deaths', 'femininity_rating', 'fem_binary', 'femininity_mturk',
    'category_ss', 'wind_mph', 'min_pressure', 'year',
    'damage_norm_2015'
]

df = _df2[cols].copy()

# Some rows may have missing values
# Use deaths >=0 and non-missing in key variables

df = df.replace([np.inf, -np.inf], np.nan)

# Derived variables
# Log-transform deaths to reduce skew

df['log_deaths'] = np.log1p(df['deaths'])

# Basic correlations
corr_pearson = df[['femininity_rating', 'log_deaths']].corr(method='pearson').iloc[0,1]
corr_spearman = df[['femininity_rating', 'deaths']].corr(method='spearman').iloc[0,1]

# Regression with controls
# Controls: category (storm strength), wind speed, min pressure, year (time trend), damage (severity)
X = df[['femininity_rating', 'category_ss', 'wind_mph', 'min_pressure', 'year', 'damage_norm_2015']]
X = sm.add_constant(X, has_constant='add')

y = df['log_deaths']

# Drop rows with missing values for regression
reg_df = pd.concat([y, X], axis=1).dropna()

y_reg = reg_df['log_deaths']
X_reg = reg_df.drop(columns=['log_deaths'])

model = sm.OLS(y_reg, X_reg).fit(cov_type='HC3')

# Also check binary female indicator
Xb = df[['fem_binary', 'category_ss', 'wind_mph', 'min_pressure', 'year', 'damage_norm_2015']]
Xb = sm.add_constant(Xb, has_constant='add')
reg_df_b = pd.concat([df['log_deaths'], Xb], axis=1).dropna()
model_b = sm.OLS(reg_df_b['log_deaths'], reg_df_b.drop(columns=['log_deaths'])).fit(cov_type='HC3')

# Print summary stats
print('N total', len(df))
print('N reg', len(reg_df))
print('Pearson corr (fem rating vs log deaths):', corr_pearson)
print('Spearman corr (fem rating vs deaths):', corr_spearman)

print('\nOLS log(deaths) ~ femininity + controls (HC3):')
print(model.summary())

print('\nOLS log(deaths) ~ female (binary) + controls (HC3):')
print(model_b.summary())

