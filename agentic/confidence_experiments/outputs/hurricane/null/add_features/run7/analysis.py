import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns used are numeric
for col in ['masfem', 'masfem_mturk', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'ndam', 'ndam15']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Outcome: log deaths
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Helper to run OLS and return key stats

def run_ols(y, X, add_const=True):
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    if add_const:
        X_clean = sm.add_constant(X_clean, has_constant='add')
    model = sm.OLS(y_clean, X_clean).fit(cov_type='HC3')
    return model

results = {}

# Model 1: log_deaths ~ masfem
if 'masfem' in _df.columns:
    m1 = run_ols(_df['log_deaths'], _df[['masfem']])
    results['m1_masfem'] = {
        'coef': m1.params.get('masfem', np.nan),
        'pval': m1.pvalues.get('masfem', np.nan),
        'r2': m1.rsquared,
        'n': int(m1.nobs),
    }

# Model 2: log_deaths ~ masfem + intensity controls
controls = [c for c in ['wind', 'min', 'category', 'ndam15'] if c in _df.columns]
if 'masfem' in _df.columns and controls:
    m2 = run_ols(_df['log_deaths'], _df[['masfem'] + controls])
    results['m2_masfem_controls'] = {
        'coef': m2.params.get('masfem', np.nan),
        'pval': m2.pvalues.get('masfem', np.nan),
        'r2': m2.rsquared,
        'n': int(m2.nobs),
    }

# Model 3: log_deaths ~ gender_mf
if 'gender_mf' in _df.columns:
    m3 = run_ols(_df['log_deaths'], _df[['gender_mf']])
    results['m3_gender_mf'] = {
        'coef': m3.params.get('gender_mf', np.nan),
        'pval': m3.pvalues.get('gender_mf', np.nan),
        'r2': m3.rsquared,
        'n': int(m3.nobs),
    }

# Model 4: log_deaths ~ gender_mf + controls
if 'gender_mf' in _df.columns and controls:
    m4 = run_ols(_df['log_deaths'], _df[['gender_mf'] + controls])
    results['m4_gender_mf_controls'] = {
        'coef': m4.params.get('gender_mf', np.nan),
        'pval': m4.pvalues.get('gender_mf', np.nan),
        'r2': m4.rsquared,
        'n': int(m4.nobs),
    }

# Correlations
corrs = {}
for x in ['masfem', 'masfem_mturk', 'gender_mf']:
    if x in _df.columns:
        corr = _df[[x, 'log_deaths']].corr().iloc[0,1]
        corrs[x] = float(corr)
results['correlations'] = corrs

print(json.dumps(results, indent=2))
