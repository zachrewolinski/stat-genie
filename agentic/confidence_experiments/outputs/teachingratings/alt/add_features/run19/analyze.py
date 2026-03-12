import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleanup: drop rows with missing beauty or eval
base = df[['beauty', 'eval']].copy()
base = base.dropna()

# Correlation
corr = base['beauty'].corr(base['eval'])

# Simple OLS
ols_simple = smf.ols('eval ~ beauty', data=df).fit()

# Build a more controlled model if variables exist
# We'll include commonly used covariates in the paper when available.
covariates = []
for col in ['age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students']:
    if col in df.columns:
        covariates.append(col)

# Create formula with categorical handling for object columns
formula = 'eval ~ beauty'
if covariates:
    formula += ' + ' + ' + '.join([f'C({c})' if df[c].dtype == 'object' else c for c in covariates])

ols_ctrl = smf.ols(formula, data=df).fit()

# Collect results
result = {
    'n_total': int(df.shape[0]),
    'n_used_simple': int(ols_simple.nobs),
    'corr_beauty_eval': float(corr),
    'simple_coef': float(ols_simple.params['beauty']),
    'simple_pvalue': float(ols_simple.pvalues['beauty']),
    'simple_ci': [float(x) for x in ols_simple.conf_int().loc['beauty'].tolist()],
    'ctrl_formula': formula,
    'ctrl_coef': float(ols_ctrl.params['beauty']),
    'ctrl_pvalue': float(ols_ctrl.pvalues['beauty']),
    'ctrl_ci': [float(x) for x in ols_ctrl.conf_int().loc['beauty'].tolist()],
    'ctrl_nobs': int(ols_ctrl.nobs),
    'ctrl_r2': float(ols_ctrl.rsquared),
    'simple_r2': float(ols_simple.rsquared),
}

print(json.dumps(result, indent=2))
