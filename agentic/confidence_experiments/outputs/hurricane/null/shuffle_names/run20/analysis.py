import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename for clarity (based on info.json descriptions)
# year: storm year; name: deaths; category: femininity rating (1-11);
# ndam15: min pressure; gender_mf: Saffir-Simpson category; year (col) = max wind speed
# ind: mturk femininity rating; masfem_mturk: binary gender

# Create analysis variables

df = df.copy()
df['deaths'] = df['name']
df['log_deaths'] = np.log1p(df['deaths'])
df['fem_rating'] = df['category']
df['fem_rating_mturk'] = df['ind']
df['fem_binary'] = df['masfem_mturk']

df['year_storm'] = df['wind']
df['wind_speed'] = df['year']
df['min_pressure'] = df['ndam15']
df['ss_category'] = df['gender_mf']

# Drop rows with missing key values
key_cols = ['deaths', 'log_deaths', 'fem_rating', 'wind_speed', 'min_pressure', 'ss_category']
base = df.dropna(subset=key_cols).copy()

# Simple correlations
corr_fem_deaths = base['fem_rating'].corr(base['deaths'])
corr_fem_log = base['fem_rating'].corr(base['log_deaths'])

# OLS regression on log deaths
model1 = smf.ols('log_deaths ~ fem_rating', data=base).fit()
model2 = smf.ols('log_deaths ~ fem_rating + wind_speed + min_pressure + ss_category + year_storm', data=base).fit()

# Alternative using mturk rating
model3 = smf.ols('log_deaths ~ fem_rating_mturk + wind_speed + min_pressure + ss_category + year_storm', data=base).fit()

# Binary female indicator
model4 = smf.ols('log_deaths ~ fem_binary + wind_speed + min_pressure + ss_category + year_storm', data=base).fit()

# Negative binomial on deaths (count)
# Add constant for GLM
glm_nb = smf.glm('deaths ~ fem_rating + wind_speed + min_pressure + ss_category + year_storm',
                 data=base, family=sm.families.NegativeBinomial()).fit()

print('N:', len(base))
print('corr fem vs deaths:', corr_fem_deaths)
print('corr fem vs log deaths:', corr_fem_log)
print('\nOLS log deaths ~ fem_rating')
print(model1.summary().tables[1])
print('\nOLS log deaths ~ fem_rating + controls')
print(model2.summary().tables[1])
print('\nOLS log deaths ~ fem_rating_mturk + controls')
print(model3.summary().tables[1])
print('\nOLS log deaths ~ fem_binary + controls')
print(model4.summary().tables[1])
print('\nGLM NB deaths ~ fem_rating + controls')
print(glm_nb.summary().tables[1])

# Save key stats for later
stats = {
    'N': int(len(base)),
    'corr_fem_deaths': float(corr_fem_deaths),
    'corr_fem_log': float(corr_fem_log),
    'model1_fem_coef': float(model1.params['fem_rating']),
    'model1_fem_p': float(model1.pvalues['fem_rating']),
    'model2_fem_coef': float(model2.params['fem_rating']),
    'model2_fem_p': float(model2.pvalues['fem_rating']),
    'model3_fem_coef': float(model3.params['fem_rating_mturk']),
    'model3_fem_p': float(model3.pvalues['fem_rating_mturk']),
    'model4_fem_coef': float(model4.params['fem_binary']),
    'model4_fem_p': float(model4.pvalues['fem_binary']),
    'glm_nb_fem_coef': float(glm_nb.params['fem_rating']),
    'glm_nb_fem_p': float(glm_nb.pvalues['fem_rating']),
}

import json
with open('analysis_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
