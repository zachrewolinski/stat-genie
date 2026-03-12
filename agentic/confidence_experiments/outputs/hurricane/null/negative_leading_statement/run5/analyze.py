import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'hurricane.csv'
_df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem','masfem_mturk','gender_mf','alldeaths','category','wind','min','ndam','ndam15','year']
for c in num_cols:
    if c in _df.columns:
        _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Drop rows with missing key variables for modeling
# Use masfem as femininity measure
analysis = _df.dropna(subset=['masfem','alldeaths','category','wind','min','year']).copy()

analysis['log_deaths'] = np.log1p(analysis['alldeaths'])

# Standardize predictors for effect comparison
for c in ['masfem','category','wind','min','year']:
    analysis[c+'_z'] = (analysis[c] - analysis[c].mean()) / analysis[c].std(ddof=0)

results = {}

# Simple correlation
corr = stats.pearsonr(analysis['masfem'], analysis['log_deaths'])
results['corr_masfem_logdeaths'] = {'r': corr.statistic, 'p': corr.pvalue, 'n': len(analysis)}

# OLS with controls
formula = 'log_deaths ~ masfem_z + category_z + wind_z + min_z + year_z'
ols = smf.ols(formula, data=analysis).fit(cov_type='HC3')
results['ols_controls'] = {
    'coef_masfem_z': ols.params['masfem_z'],
    'se_masfem_z': ols.bse['masfem_z'],
    'p_masfem_z': ols.pvalues['masfem_z'],
    'r2': ols.rsquared,
    'n': int(ols.nobs)
}

# Alternative: using gender_mf (binary) instead of masfem
analysis_g = _df.dropna(subset=['gender_mf','alldeaths','category','wind','min','year']).copy()
analysis_g['log_deaths'] = np.log1p(analysis_g['alldeaths'])
for c in ['category','wind','min','year']:
    analysis_g[c+'_z'] = (analysis_g[c] - analysis_g[c].mean()) / analysis_g[c].std(ddof=0)

formula_g = 'log_deaths ~ gender_mf + category_z + wind_z + min_z + year_z'
ols_g = smf.ols(formula_g, data=analysis_g).fit(cov_type='HC3')
results['ols_gender_controls'] = {
    'coef_gender_mf': ols_g.params['gender_mf'],
    'se_gender_mf': ols_g.bse['gender_mf'],
    'p_gender_mf': ols_g.pvalues['gender_mf'],
    'r2': ols_g.rsquared,
    'n': int(ols_g.nobs)
}

# Count model (Poisson) with log link
# Use robust SEs
poisson = smf.glm('alldeaths ~ masfem_z + category_z + wind_z + min_z + year_z',
                  data=analysis, family=sm.families.Poisson()).fit(cov_type='HC3')
results['poisson_controls'] = {
    'coef_masfem_z': poisson.params['masfem_z'],
    'se_masfem_z': poisson.bse['masfem_z'],
    'p_masfem_z': poisson.pvalues['masfem_z'],
    'n': int(poisson.nobs)
}

# Negative binomial (using GLM) to handle overdispersion
nb = smf.glm('alldeaths ~ masfem_z + category_z + wind_z + min_z + year_z',
             data=analysis, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
results['neg_bin_controls'] = {
    'coef_masfem_z': nb.params['masfem_z'],
    'se_masfem_z': nb.bse['masfem_z'],
    'p_masfem_z': nb.pvalues['masfem_z'],
    'n': int(nb.nobs)
}

# Save results to json for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
