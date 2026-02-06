import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Rename columns to clearer semantic names based on inspection
# ndam: row id
# wind: year of hurricane
# alldeaths: hurricane name (string)
# category: femininity rating (1-11)
# ndam15: minimum pressure at landfall
# masfem_mturk: binary female indicator (0 male, 1 female)
# gender_mf: Saffir-Simpson category (1-5)
# name: deaths
# elapsedyrs: normalized damages (2013)
# masfem: years elapsed since hurricane
# min: data source
# ind: MTurk femininity rating (1-11)
# year: wind speed at landfall
# source: normalized damages (2015)

df = _df.copy()

# Outcome: deaths (highly skewed)
df['deaths'] = df['name']

df['log_deaths'] = np.log1p(df['deaths'])

# Key predictors
fem_rating_main = 'category'
fem_rating_alt = 'ind'

# Controls for storm intensity
controls = ['year', 'gender_mf', 'ndam15']  # wind speed, category, min pressure

# Drop rows with missing values in any model variables
model_vars_main = ['log_deaths', fem_rating_main] + controls
model_vars_alt = ['log_deaths', fem_rating_alt] + controls
model_vars_bin = ['log_deaths', 'masfem_mturk'] + controls

_df_main = df[model_vars_main].dropna()
_df_alt = df[model_vars_alt].dropna()
_df_bin = df[model_vars_bin].dropna()

# Fit OLS models
X_main = sm.add_constant(_df_main[[fem_rating_main] + controls])
model_main = sm.OLS(_df_main['log_deaths'], X_main).fit()

X_alt = sm.add_constant(_df_alt[[fem_rating_alt] + controls])
model_alt = sm.OLS(_df_alt['log_deaths'], X_alt).fit()

X_bin = sm.add_constant(_df_bin[['masfem_mturk'] + controls])
model_bin = sm.OLS(_df_bin['log_deaths'], X_bin).fit()

# Simple correlations (uncontrolled)
corr_main = df[['deaths', fem_rating_main]].dropna().corr().iloc[0,1]
corr_alt = df[['deaths', fem_rating_alt]].dropna().corr().iloc[0,1]

# Collect results
results = {
    'corr_deaths_fem_rating_main': float(corr_main),
    'corr_deaths_fem_rating_alt': float(corr_alt),
    'model_main_coef_fem': float(model_main.params[fem_rating_main]),
    'model_main_pvalue_fem': float(model_main.pvalues[fem_rating_main]),
    'model_alt_coef_fem': float(model_alt.params[fem_rating_alt]),
    'model_alt_pvalue_fem': float(model_alt.pvalues[fem_rating_alt]),
    'model_bin_coef_female': float(model_bin.params['masfem_mturk']),
    'model_bin_pvalue_female': float(model_bin.pvalues['masfem_mturk']),
    'n_main': int(_df_main.shape[0]),
    'n_alt': int(_df_alt.shape[0]),
    'n_bin': int(_df_bin.shape[0]),
}

print(results)
