import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('hurricane.csv')

# Keep relevant columns and drop missing
cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'year']

# Some datasets might use different case; ensure columns exist
missing = [c for c in cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# Prepare data
sub = df[cols].copy()

# log transform deaths for skew (add 1 to allow zeros)
sub['log_deaths'] = np.log1p(sub['alldeaths'])

# Simple correlations
corr = sub[['masfem', 'log_deaths', 'alldeaths']].corr()

# Regression: log deaths on femininity + intensity controls
X = sub[['masfem', 'wind', 'min', 'category', 'year']]
X = sm.add_constant(X)
model = sm.OLS(sub['log_deaths'], X).fit(cov_type='HC3')

# Alternative: binary gender indicator
X2 = sub[['gender_mf', 'wind', 'min', 'category', 'year']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(sub['log_deaths'], X2).fit(cov_type='HC3')

# Standardized coefficients for masfem (approx)
# standardize variables
std = (sub[['masfem', 'wind', 'min', 'category', 'year', 'log_deaths']] - sub[['masfem', 'wind', 'min', 'category', 'year', 'log_deaths']].mean()) / sub[['masfem', 'wind', 'min', 'category', 'year', 'log_deaths']].std()
X_std = sm.add_constant(std[['masfem', 'wind', 'min', 'category', 'year']])
model_std = sm.OLS(std['log_deaths'], X_std).fit(cov_type='HC3')

results = {
    'n': len(sub),
    'corr_masfem_logdeaths': float(corr.loc['masfem', 'log_deaths']),
    'corr_masfem_deaths': float(corr.loc['masfem', 'alldeaths']),
    'ols_masfem_coef': float(model.params['masfem']),
    'ols_masfem_p': float(model.pvalues['masfem']),
    'ols_masfem_ci': [float(x) for x in model.conf_int().loc['masfem']],
    'ols_gender_coef': float(model2.params['gender_mf']),
    'ols_gender_p': float(model2.pvalues['gender_mf']),
    'ols_gender_ci': [float(x) for x in model2.conf_int().loc['gender_mf']],
    'ols_std_masfem_coef': float(model_std.params['masfem']),
    'ols_std_masfem_p': float(model_std.pvalues['masfem']),
}

print(json.dumps(results, indent=2))
