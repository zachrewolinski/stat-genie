import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns for readability
colmap = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'mas_fem_index',
    'feature5': 'min_pressure',
    'feature6': 'female_name',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_elapsed',
    'feature11': 'source',
    'feature12': 'mturk_mas_fem',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=colmap)

# Basic summary
summary = df[['mas_fem_index','mturk_mas_fem','female_name','deaths','category','max_wind','min_pressure','damage_2013','damage_2015']].describe()

# Log-transform deaths to reduce skew (add 1 for zeros)
df['log_deaths'] = np.log1p(df['deaths'])

# Correlations
corrs = df[['mas_fem_index','mturk_mas_fem','female_name','deaths','log_deaths']].corr()

# Simple OLS: log_deaths ~ mas_fem_index
model1 = smf.ols('log_deaths ~ mas_fem_index', data=df).fit()

# Controls for storm intensity (category, max_wind, min_pressure) and damage (2013) as exposure proxy
# Use damage_2013 because it's normalized to 2013 values; log-transform with +1
# Also include year to control for temporal trends in safety/preparedness

# Prepare logs
for col in ['damage_2013','damage_2015','max_wind','min_pressure']:
    # use log for skewed variables (pressure might not need log but use as-is if log not desired)
    pass

df['log_damage_2013'] = np.log1p(df['damage_2013'])

model2 = smf.ols('log_deaths ~ mas_fem_index + category + max_wind + min_pressure + log_damage_2013 + year', data=df).fit()

# Alternative: binary female name
model3 = smf.ols('log_deaths ~ female_name + category + max_wind + min_pressure + log_damage_2013 + year', data=df).fit()

# Alternative: use mturk_mas_fem
model4 = smf.ols('log_deaths ~ mturk_mas_fem + category + max_wind + min_pressure + log_damage_2013 + year', data=df).fit()

# Robust (HC3) standard errors to be safe
model2_hc3 = model2.get_robustcov_results(cov_type='HC3')
model3_hc3 = model3.get_robustcov_results(cov_type='HC3')
model4_hc3 = model4.get_robustcov_results(cov_type='HC3')

def _series_from_results(results):
    names = results.model.exog_names
    params = pd.Series(results.params, index=names)
    pvals = pd.Series(results.pvalues, index=names)
    return params.to_dict(), pvals.to_dict()

# Save key outputs
out = {
    'summary': summary.to_dict(),
    'corrs': corrs.to_dict(),
    'model1': {
        'params': model1.params.to_dict(),
        'pvalues': model1.pvalues.to_dict(),
        'r2': model1.rsquared,
        'nobs': int(model1.nobs),
    },
    'model2': {
        'params': model2.params.to_dict(),
        'pvalues': model2.pvalues.to_dict(),
        'r2': model2.rsquared,
        'nobs': int(model2.nobs),
    },
    'model2_hc3': {
        'params': _series_from_results(model2_hc3)[0],
        'pvalues': _series_from_results(model2_hc3)[1],
    },
    'model3': {
        'params': model3.params.to_dict(),
        'pvalues': model3.pvalues.to_dict(),
        'r2': model3.rsquared,
        'nobs': int(model3.nobs),
    },
    'model3_hc3': {
        'params': _series_from_results(model3_hc3)[0],
        'pvalues': _series_from_results(model3_hc3)[1],
    },
    'model4': {
        'params': model4.params.to_dict(),
        'pvalues': model4.pvalues.to_dict(),
        'r2': model4.rsquared,
        'nobs': int(model4.nobs),
    },
    'model4_hc3': {
        'params': _series_from_results(model4_hc3)[0],
        'pvalues': _series_from_results(model4_hc3)[1],
    },
}

import json
with open('analysis_results.json','w') as f:
    json.dump(out,f,indent=2)

print('done')
