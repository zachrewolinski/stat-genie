import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic checks
n_rows = _df.shape[0]

# Define variables of interest
# Research question: does instructor beauty affect teaching productivity as reflected in student instructional ratings?
# We'll treat 'beauty' as predictor and 'allstudents' as rating outcome.

# Drop rows with missing in key variables
key_cols = ['beauty', 'allstudents']
_df_key = _df.dropna(subset=key_cols).copy()

# Pearson correlation
corr, corr_p = stats.pearsonr(_df_key['beauty'], _df_key['allstudents'])

# Simple OLS
X_simple = sm.add_constant(_df_key['beauty'])
model_simple = sm.OLS(_df_key['allstudents'], X_simple).fit(cov_type='HC3')

# Multiple OLS with available covariates
# Build design matrix with numeric and one-hot encoded categorical variables (excluding outcome and beauty)
# Identify categorical columns by dtype object
categorical_cols = [c for c in _df.columns if _df[c].dtype == 'object']
# Exclude outcome and predictor if they are categorical (they are not)
exclude_cols = set(['allstudents'])

# Numeric columns excluding outcome
numeric_cols = [c for c in _df.columns if _df[c].dtype != 'object' and c not in exclude_cols]
# Ensure beauty included as numeric

# Prepare data
_df_model = _df.dropna(subset=['allstudents']).copy()

# One-hot encode categoricals
if categorical_cols:
    dummies = pd.get_dummies(_df_model[categorical_cols], drop_first=True)
else:
    dummies = pd.DataFrame(index=_df_model.index)

X_multi = pd.concat([
    _df_model[numeric_cols],
    dummies
], axis=1)

# Remove outcome if accidentally included
if 'allstudents' in X_multi.columns:
    X_multi = X_multi.drop(columns=['allstudents'])

# Add constant
X_multi = sm.add_constant(X_multi, has_constant='add')

# Fit model
model_multi = sm.OLS(_df_model['allstudents'], X_multi).fit(cov_type='HC3')

# Extract beauty coefficient in multiple model
beauty_coef = model_multi.params.get('beauty', np.nan)
beauty_p = model_multi.pvalues.get('beauty', np.nan)

# Save summary stats to json for inspection
summary = {
    'n_rows': int(n_rows),
    'n_used_key': int(_df_key.shape[0]),
    'corr': float(corr),
    'corr_p': float(corr_p),
    'simple_coef': float(model_simple.params['beauty']),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_r2': float(model_simple.rsquared),
    'multi_coef': float(beauty_coef),
    'multi_p': float(beauty_p),
    'multi_r2': float(model_multi.rsquared),
    'multi_adj_r2': float(model_multi.rsquared_adj),
}

print(json.dumps(summary, indent=2))
