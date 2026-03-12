import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns
num_cols = ['masfem','masfem_mturk','alldeaths','wind','min','category','year','ndam','ndam15','elapsedyrs']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Outcome: fatalities
# Use log1p to handle zeros

df['log_deaths'] = np.log1p(df['alldeaths'])

# Standardize some controls? We'll just use raw; OLS with robust SE.

# Model 1: simple association
model1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# Model 2: controls for storm intensity (wind, min pressure, category)
model2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC3')

# Model 3: add year trend (or elapsedyrs)
model3 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')

# Model 4: use MTurk rating
model4 = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=df).fit(cov_type='HC3')

# Poisson regression with robust SE for count outcome
# Avoid zeros issues: Poisson can handle zeros.
poisson = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Summaries

def extract(model, var):
    return {
        'coef': float(model.params.get(var, np.nan)),
        'se': float(model.bse.get(var, np.nan)),
        'p': float(model.pvalues.get(var, np.nan))
    }

results = {
    'n': int(df.shape[0]),
    'model1': extract(model1, 'masfem'),
    'model2': extract(model2, 'masfem'),
    'model3': extract(model3, 'masfem'),
    'model4_mturk': extract(model4, 'masfem_mturk'),
    'poisson': extract(poisson, 'masfem'),
}

# Additional descriptive: correlation
results['corr_masfem_logdeaths'] = float(df['masfem'].corr(df['log_deaths']))
results['corr_masfem_deaths'] = float(df['masfem'].corr(df['alldeaths']))

# Write results for later use
import json
with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
