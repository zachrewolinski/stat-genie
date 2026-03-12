import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem','masfem_mturk','alldeaths','wind','min','category','ndam','ndam15','elapsedyrs','gender_mf','year']
for c in num_cols:
    if c in _df.columns:
        _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Outcome transforms
_df['log_deaths'] = np.log(_df['alldeaths'] + 1)

# Core covariates for intensity
# Use wind, min pressure, category; these are common severity controls

results = {}

# Correlations
results['corr_masfem_alldeaths'] = _df[['masfem','alldeaths']].corr().iloc[0,1]
results['corr_masfem_log_deaths'] = _df[['masfem','log_deaths']].corr().iloc[0,1]
results['corr_masfem_mturk_log_deaths'] = _df[['masfem_mturk','log_deaths']].corr().iloc[0,1]

# Simple OLS
ols1 = smf.ols('log_deaths ~ masfem', data=_df).fit()
results['ols1'] = {'coef': ols1.params['masfem'], 'p': ols1.pvalues['masfem'], 'n': int(ols1.nobs), 'r2': ols1.rsquared}

# OLS with severity controls
ols2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=_df).fit()
results['ols2'] = {
    'coef': ols2.params['masfem'],
    'p': ols2.pvalues['masfem'],
    'n': int(ols2.nobs),
    'r2': ols2.rsquared,
}

# Add year to account for improvements in warning systems over time
ols3 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=_df).fit()
results['ols3'] = {
    'coef': ols3.params['masfem'],
    'p': ols3.pvalues['masfem'],
    'n': int(ols3.nobs),
    'r2': ols3.rsquared,
}

# Use gender_mf (binary) as alternative to masfem
ols4 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=_df).fit()
results['ols4'] = {
    'coef': ols4.params['gender_mf'],
    'p': ols4.pvalues['gender_mf'],
    'n': int(ols4.nobs),
    'r2': ols4.rsquared,
}

# Alternative femininity measure (MTurk)
ols5 = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=_df).fit()
results['ols5'] = {
    'coef': ols5.params['masfem_mturk'],
    'p': ols5.pvalues['masfem_mturk'],
    'n': int(ols5.nobs),
    'r2': ols5.rsquared,
}

# Poisson regression for count outcome
# Use robust SE to handle overdispersion
poisson1 = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=_df,
                   family=sm.families.Poisson()).fit(cov_type='HC3')
results['poisson1'] = {
    'coef': poisson1.params['masfem'],
    'p': poisson1.pvalues['masfem'],
    'n': int(poisson1.nobs),
}

poisson2 = smf.glm('alldeaths ~ masfem_mturk + wind + min + category + year', data=_df,
                   family=sm.families.Poisson()).fit(cov_type='HC3')
results['poisson2'] = {
    'coef': poisson2.params['masfem_mturk'],
    'p': poisson2.pvalues['masfem_mturk'],
    'n': int(poisson2.nobs),
}

# Negative binomial regression (more appropriate for overdispersion)
# Use statsmodels discrete NB2
try:
    nb1 = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=_df,
                 family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    results['nb1'] = {
        'coef': nb1.params['masfem'],
        'p': nb1.pvalues['masfem'],
        'n': int(nb1.nobs),
    }
    nb2 = smf.glm('alldeaths ~ masfem_mturk + wind + min + category + year', data=_df,
                 family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    results['nb2'] = {
        'coef': nb2.params['masfem_mturk'],
        'p': nb2.pvalues['masfem_mturk'],
        'n': int(nb2.nobs),
    }
except Exception as e:
    results['nb1_error'] = str(e)

# Save results for inspection
with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
