import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic data prep
# Ensure numeric columns are numeric
numeric_cols = ['masfem','masfem_mturk','gender_mf','category','wind','min','alldeaths','ndam','ndam15','year']
for c in numeric_cols:
    if c in _df.columns:
        _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Log transform deaths (log1p to handle zeros)
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Descriptives
summary = _df[['masfem','gender_mf','category','wind','min','alldeaths','log_deaths']].describe().T

# Correlations with femininity
corrs = _df[['masfem','alldeaths','log_deaths','category','wind','min']].corr()

# Simple OLS: log deaths ~ masfem
m1 = smf.ols('log_deaths ~ masfem', data=_df).fit(cov_type='HC3')

# OLS with intensity controls
m2 = smf.ols('log_deaths ~ masfem + category + wind + min', data=_df).fit(cov_type='HC3')

# OLS with year (temporal trend) + controls
m3 = smf.ols('log_deaths ~ masfem + category + wind + min + year', data=_df).fit(cov_type='HC3')

# Alternative: gender binary instead of masfem
m4 = smf.ols('log_deaths ~ gender_mf + category + wind + min + year', data=_df).fit(cov_type='HC3')

# Poisson regression for counts with log link (use alldeaths) with controls
# Add small constant to avoid issues with zeros in poisson? not needed.
poisson = smf.glm('alldeaths ~ masfem + category + wind + min + year',
                  data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Prepare output summary stats for key models

def model_table(model, name):
    return {
        'model': name,
        'params': model.params.to_dict(),
        'bse': model.bse.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'rsq': getattr(model, 'rsquared', None),
        'n': int(model.nobs)
    }

results = {
    'summary': summary,
    'corrs': corrs,
    'models': [
        model_table(m1, 'ols_log_deaths_masfem'),
        model_table(m2, 'ols_log_deaths_masfem_controls'),
        model_table(m3, 'ols_log_deaths_masfem_controls_year'),
        model_table(m4, 'ols_log_deaths_gender_controls_year'),
        model_table(poisson, 'poisson_deaths_masfem_controls_year')
    ]
}

# Print concise
print('Rows:', len(_df))
print('\nCorrelation (masfem with log_deaths):', corrs.loc['masfem','log_deaths'])
print('\nOLS log_deaths ~ masfem (HC3):')
print(m1.summary().tables[1])
print('\nOLS log_deaths ~ masfem + category + wind + min (HC3):')
print(m2.summary().tables[1])
print('\nOLS log_deaths ~ masfem + category + wind + min + year (HC3):')
print(m3.summary().tables[1])
print('\nOLS log_deaths ~ gender_mf + category + wind + min + year (HC3):')
print(m4.summary().tables[1])
print('\nPoisson deaths ~ masfem + category + wind + min + year (HC3):')
print(poisson.summary().tables[1])

# Save detailed tables for inspection if needed
summary.to_csv('analysis_summary.csv')
corrs.to_csv('analysis_corrs.csv')

import json
with open('analysis_models.json','w') as f:
    json.dump(results, f, indent=2, default=str)
