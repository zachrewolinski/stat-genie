import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

path = 'hurricane.csv'

df = pd.read_csv(path)

# Map columns to intended variables based on metadata values
storm_year = df['wind']
name = df['alldeaths']
# femininity index (1-11)
fem_index = df['category']
# minimum pressure
min_pressure = df['ndam15']
# binary female name indicator
female_binary = df['masfem_mturk']
# Saffir-Simpson category
storm_cat = df['gender_mf']
# deaths
fatalities = df['name']
# max wind speed
max_wind = df['year']

# log transform deaths (skewed)
log_deaths = np.log1p(fatalities)

# Helper to run OLS and return summary stats

def run_ols(y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit()
    return model

results = {}

# Bivariate: fem index vs log deaths
model_biv = run_ols(log_deaths, pd.DataFrame({'fem_index': fem_index}))
results['biv_fem'] = {
    'coef': model_biv.params['fem_index'],
    'pval': model_biv.pvalues['fem_index'],
    'r2': model_biv.rsquared,
}

# Multivariate controls
controls = pd.DataFrame({
    'fem_index': fem_index,
    'storm_cat': storm_cat,
    'max_wind': max_wind,
    'min_pressure': min_pressure,
    'storm_year': storm_year,
})
model_multi = run_ols(log_deaths, controls)
results['multi_fem'] = {
    'coef': model_multi.params['fem_index'],
    'pval': model_multi.pvalues['fem_index'],
    'r2': model_multi.rsquared,
}

# Binary female indicator model
controls_bin = pd.DataFrame({
    'female_binary': female_binary,
    'storm_cat': storm_cat,
    'max_wind': max_wind,
    'min_pressure': min_pressure,
    'storm_year': storm_year,
})
model_bin = run_ols(log_deaths, controls_bin)
results['multi_binary'] = {
    'coef': model_bin.params['female_binary'],
    'pval': model_bin.pvalues['female_binary'],
    'r2': model_bin.rsquared,
}

# Correlations
results['corr'] = {
    'fem_index_log_deaths': np.corrcoef(fem_index, log_deaths)[0,1],
    'female_binary_log_deaths': np.corrcoef(female_binary, log_deaths)[0,1],
}

# Basic descriptive stats for deaths
results['deaths'] = {
    'count': int(fatalities.count()),
    'mean': float(fatalities.mean()),
    'median': float(fatalities.median()),
    'max': int(fatalities.max()),
}

print(json.dumps(results, indent=2))
