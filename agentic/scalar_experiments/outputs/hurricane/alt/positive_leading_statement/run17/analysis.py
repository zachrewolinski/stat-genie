import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns
num_cols = ['masfem', 'masfem_mturk', 'wind', 'min', 'category', 'alldeaths', 'ndam', 'ndam15', 'elapsedyrs', 'gender_mf']
for c in num_cols:
    if c in _df.columns:
        _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Create log deaths and log damages
_df['log_deaths'] = np.log1p(_df['alldeaths'])
_df['log_ndam15'] = np.log1p(_df['ndam15'])
_df['log_ndam'] = np.log1p(_df['ndam'])

# Drop rows with missing key variables
key_vars = ['log_deaths', 'masfem', 'wind', 'min', 'category', 'log_ndam15']
_df_model = _df.dropna(subset=key_vars).copy()

results = {}

# Model 1: OLS with robust SEs
formula1 = 'log_deaths ~ masfem + wind + min + category + log_ndam15'
model1 = smf.ols(formula1, data=_df_model).fit(cov_type='HC3')
results['ols'] = {
    'n': int(model1.nobs),
    'coef_masfem': float(model1.params.get('masfem', np.nan)),
    'se_masfem': float(model1.bse.get('masfem', np.nan)),
    'p_masfem': float(model1.pvalues.get('masfem', np.nan)),
    'r2': float(model1.rsquared),
}

# Model 2: OLS with masfem_mturk (if available)
if 'masfem_mturk' in _df.columns:
    key_vars2 = ['log_deaths', 'masfem_mturk', 'wind', 'min', 'category', 'log_ndam15']
    _df_model2 = _df.dropna(subset=key_vars2).copy()
    formula2 = 'log_deaths ~ masfem_mturk + wind + min + category + log_ndam15'
    model2 = smf.ols(formula2, data=_df_model2).fit(cov_type='HC3')
    results['ols_mturk'] = {
        'n': int(model2.nobs),
        'coef_masfem_mturk': float(model2.params.get('masfem_mturk', np.nan)),
        'se_masfem_mturk': float(model2.bse.get('masfem_mturk', np.nan)),
        'p_masfem_mturk': float(model2.pvalues.get('masfem_mturk', np.nan)),
        'r2': float(model2.rsquared),
    }

# Model 3: Negative binomial on deaths (count) with robust SE
# Use GLM with Negative Binomial to handle over-dispersion
try:
    formula3 = 'alldeaths ~ masfem + wind + min + category + log_ndam15'
    model3 = smf.glm(formula3, data=_df_model, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    results['negbin'] = {
        'n': int(model3.nobs),
        'coef_masfem': float(model3.params.get('masfem', np.nan)),
        'se_masfem': float(model3.bse.get('masfem', np.nan)),
        'p_masfem': float(model3.pvalues.get('masfem', np.nan)),
    }
except Exception as e:
    results['negbin_error'] = str(e)

# Simple bivariate correlation
corr = _df[['masfem', 'alldeaths']].corr().iloc[0,1]
results['corr_masfem_deaths'] = float(corr)

print(json.dumps(results, indent=2))
