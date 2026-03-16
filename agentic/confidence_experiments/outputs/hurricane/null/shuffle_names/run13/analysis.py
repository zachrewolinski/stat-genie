import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('hurricane.csv')

# Load info to map semantics (optional for clarity)
with open('info.json', 'r') as f:
    info = json.load(f)

# For clarity map columns to descriptions
col_desc = {f['column']: f['properties'].get('description', '') for f in info['data_desc']['fields']}

# Basic overview
summary = _df.describe(include='all')
print('Columns:', list(_df.columns))
print(summary)

# Identify key variables based on descriptions
# Femininity measures
fem_rating = 'category'  # masculinity-femininity index 1-11
fem_binary = 'masfem_mturk'  # 0/1
fem_mturk = 'ind'  # mturk rating 1-11 maybe

# Outcome: total deaths
outcome = 'name'

# Controls for storm severity
controls = ['gender_mf', 'year', 'ndam15']  # category (Saffir-Simpson), max wind speed, min pressure

# Some rows might have missing outcome
_df['name'] = pd.to_numeric(_df['name'], errors='coerce')

# Also check missingness
print('Missing values in outcome:', _df[outcome].isna().sum())

# Prepare dataset for regression: drop rows with missing outcome or predictors
model_cols = [outcome, fem_rating, fem_binary, fem_mturk] + controls
model_df = _df[model_cols].copy()

# Convert to numeric
for col in model_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

# Drop missing
model_df = model_df.dropna()
print('Rows for model:', len(model_df))

# Use log1p deaths to reduce skew
model_df['log_deaths'] = np.log1p(model_df[outcome])

# Simple correlations
print('Correlations with log_deaths:')
for var in [fem_rating, fem_binary, fem_mturk]:
    r, p = stats.pearsonr(model_df[var], model_df['log_deaths'])
    print(var, r, p)

# Regression models
results = {}

def run_ols(y, X, name):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    results[name] = model
    print('\n', name)
    print(model.summary())

# Model 1: log_deaths ~ fem_rating
run_ols(model_df['log_deaths'], model_df[[fem_rating]], 'log_deaths_fem_rating')

# Model 2: log_deaths ~ fem_rating + controls
run_ols(model_df['log_deaths'], model_df[[fem_rating] + controls], 'log_deaths_fem_rating_controls')

# Model 3: log_deaths ~ fem_binary + controls
run_ols(model_df['log_deaths'], model_df[[fem_binary] + controls], 'log_deaths_fem_binary_controls')

# Model 4: log_deaths ~ fem_mturk + controls
run_ols(model_df['log_deaths'], model_df[[fem_mturk] + controls], 'log_deaths_fem_mturk_controls')

# Optional: negative binomial on deaths
# Using GLM with Negative Binomial
for var, name in [(fem_rating, 'nb_fem_rating_controls'), (fem_binary, 'nb_fem_binary_controls'), (fem_mturk, 'nb_fem_mturk_controls')]:
    X = sm.add_constant(model_df[[var] + controls])
    y = model_df[outcome]
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    results[name] = nb_model
    print('\n', name)
    print(nb_model.summary())

