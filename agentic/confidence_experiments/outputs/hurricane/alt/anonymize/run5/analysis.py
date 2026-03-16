import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('hurricane.csv')

# Define key variables
fem_index = 'feature4'  # masculinity-femininity index
female_binary = 'feature6'

deaths = 'feature8'
min_pressure = 'feature5'
category = 'feature7'
wind_speed = 'feature13'
year = 'feature2'

# Prepare data

df = df.copy()

# log1p deaths to reduce skew

df['log_deaths'] = np.log1p(df[deaths])

# Basic correlation
corr = df[[fem_index, deaths, 'log_deaths']].corr().loc[fem_index]

# Model 1: log deaths ~ fem index
X1 = sm.add_constant(df[[fem_index]])
model1 = sm.OLS(df['log_deaths'], X1, missing='drop').fit(cov_type='HC3')

# Model 2: log deaths ~ fem index + intensity controls
controls = [fem_index, min_pressure, wind_speed, category]
X2 = sm.add_constant(df[controls])
model2 = sm.OLS(df['log_deaths'], X2, missing='drop').fit(cov_type='HC3')

# Model 3: add year to control for temporal changes
controls3 = [fem_index, min_pressure, wind_speed, category, year]
X3 = sm.add_constant(df[controls3])
model3 = sm.OLS(df['log_deaths'], X3, missing='drop').fit(cov_type='HC3')

# Model 4: binary gender instead of continuous index
controls4 = [female_binary, min_pressure, wind_speed, category, year]
X4 = sm.add_constant(df[controls4])
model4 = sm.OLS(df['log_deaths'], X4, missing='drop').fit(cov_type='HC3')


def summarize_model(m, coef_name):
    return {
        'coef': float(m.params.get(coef_name, np.nan)),
        'se': float(m.bse.get(coef_name, np.nan)),
        't': float(m.tvalues.get(coef_name, np.nan)),
        'p': float(m.pvalues.get(coef_name, np.nan)),
        'n': int(m.nobs),
        'r2': float(m.rsquared),
    }

results = {
    'corr_fem_deaths': float(corr[deaths]),
    'corr_fem_log_deaths': float(corr['log_deaths']),
    'model1': summarize_model(model1, fem_index),
    'model2': summarize_model(model2, fem_index),
    'model3': summarize_model(model3, fem_index),
    'model4_female_binary': summarize_model(model4, female_binary),
}

print(json.dumps(results, indent=2))
