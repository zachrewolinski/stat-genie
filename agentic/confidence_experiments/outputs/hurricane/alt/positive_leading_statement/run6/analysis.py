import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns
for col in ['masfem','masfem_mturk','min','wind','category','alldeaths','ndam','ndam15','gender_mf']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Drop rows with missing key variables
key_cols = ['alldeaths','masfem','wind','min','category']
_df = _df.dropna(subset=key_cols)

# Create log deaths
_df['log_deaths'] = np.log1p(_df['alldeaths'])

print('Rows:', len(_df))
print('Deaths summary:', _df['alldeaths'].describe())
print('Masfem summary:', _df['masfem'].describe())

# Correlations
corr_masfem_deaths = _df[['masfem','alldeaths']].corr().iloc[0,1]
print('Correlation masfem vs deaths:', corr_masfem_deaths)

# OLS regressions
models = {}

# Simple bivariate
models['ols_simple'] = smf.ols('log_deaths ~ masfem', data=_df).fit(cov_type='HC3')
# Add intensity controls
models['ols_intensity'] = smf.ols('log_deaths ~ masfem + wind + min + category', data=_df).fit(cov_type='HC3')
# Add damage control (ndam15)
if 'ndam15' in _df.columns:
    models['ols_intensity_damage'] = smf.ols('log_deaths ~ masfem + wind + min + category + ndam15', data=_df).fit(cov_type='HC3')

# Negative binomial GLM for counts
# Use wind, min, category as controls
try:
    models['nb'] = smf.glm('alldeaths ~ masfem + wind + min + category',
                           data=_df, family=sm.families.NegativeBinomial()).fit()
except Exception as e:
    print('NB model failed:', e)

# Binary gender model
if 'gender_mf' in _df.columns:
    models['ols_gender'] = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=_df).fit(cov_type='HC3')

# Print key coefficients
for name, model in models.items():
    if 'masfem' in model.params.index:
        coef = model.params['masfem']
        pval = model.pvalues['masfem']
        print(f"{name}: masfem coef={coef:.4f}, p={pval:.4f}")
    if 'gender_mf' in model.params.index:
        coef = model.params['gender_mf']
        pval = model.pvalues['gender_mf']
        print(f"{name}: gender_mf coef={coef:.4f}, p={pval:.4f}")

# Save results to JSON for convenience
out = {}
for name, model in models.items():
    out[name] = {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'rsquared': getattr(model, 'rsquared', None),
        'nobs': int(model.nobs),
    }

with open('analysis_results.json','w') as f:
    json.dump(out, f, indent=2)

print('Saved analysis_results.json')
