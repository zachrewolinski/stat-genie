import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'

df = pd.read_csv(path)

# Map columns to semantic variables based on value ranges
mapped = pd.DataFrame({
    'id': df['ndam'],
    'year': df['wind'],  # 1950-2012
    'name': df['alldeaths'],
    'fem_rating_coder': df['category'],  # 1-11 scale
    'min_pressure': df['ndam15'],
    'fem_binary': df['masfem_mturk'],  # 0/1
    'category_ss': df['gender_mf'],  # 1-5
    'deaths': df['name'],  # 0-1833
    'damage_2013_norm': df['elapsedyrs'],
    'years_elapsed': df['masfem'],  # 1-63
    'source': df['min'],
    'fem_rating_mturk': df['ind'],  # 1-11 scale
    'wind_speed': df['year'],  # 75-190
    'damage_2015_norm': df['source']
})

# Core variables
mapped['log_deaths'] = np.log1p(mapped['deaths'])

# Drop rows with missing in model variables
base_vars = ['log_deaths', 'fem_rating_coder', 'wind_speed', 'min_pressure', 'category_ss', 'damage_2015_norm', 'year']
model_df = mapped[base_vars].dropna()

# Models
models = {}

models['univariate_coder'] = smf.ols('log_deaths ~ fem_rating_coder', data=model_df).fit(cov_type='HC3')
models['control_coder'] = smf.ols(
    'log_deaths ~ fem_rating_coder + wind_speed + min_pressure + category_ss + damage_2015_norm + year',
    data=model_df
).fit(cov_type='HC3')

# MTurk rating variant
model_df_mturk = mapped[['log_deaths', 'fem_rating_mturk', 'wind_speed', 'min_pressure', 'category_ss', 'damage_2015_norm', 'year']].dropna()
models['univariate_mturk'] = smf.ols('log_deaths ~ fem_rating_mturk', data=model_df_mturk).fit(cov_type='HC3')
models['control_mturk'] = smf.ols(
    'log_deaths ~ fem_rating_mturk + wind_speed + min_pressure + category_ss + damage_2015_norm + year',
    data=model_df_mturk
).fit(cov_type='HC3')

# Binary gender indicator
model_df_bin = mapped[['log_deaths', 'fem_binary', 'wind_speed', 'min_pressure', 'category_ss', 'damage_2015_norm', 'year']].dropna()
models['univariate_binary'] = smf.ols('log_deaths ~ fem_binary', data=model_df_bin).fit(cov_type='HC3')
models['control_binary'] = smf.ols(
    'log_deaths ~ fem_binary + wind_speed + min_pressure + category_ss + damage_2015_norm + year',
    data=model_df_bin
).fit(cov_type='HC3')

# Summaries
results = {}
for k, m in models.items():
    coef_name = [c for c in m.params.index if 'fem_' in c]
    if coef_name:
        c = coef_name[0]
        results[k] = {
            'coef': float(m.params[c]),
            'pvalue': float(m.pvalues[c]),
            'n': int(m.nobs),
            'r2': float(m.rsquared)
        }

print('RESULTS')
for k, v in results.items():
    print(k, v)

# Save results to a JSON file for easy consumption
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
