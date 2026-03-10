import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning: drop rows with missing values in key variables
key_vars = ['eval', 'beauty']
_df_key = _df.dropna(subset=key_vars).copy()

# Simple correlation
corr = _df_key['eval'].corr(_df_key['beauty'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=_df_key).fit(cov_type='HC3')

# Multivariate OLS with common controls
controls = [
    'age',
    'C(gender)',
    'C(minority)',
    'C(native)',
    'C(tenure)',
    'C(division)',
    'C(credits)',
    'students'
]
formula = 'eval ~ beauty + ' + ' + '.join(controls)
_df_controls = _df.dropna(subset=['eval', 'beauty', 'age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students']).copy()
model_controls = smf.ols(formula, data=_df_controls).fit(cov_type='HC3')

# Extract key stats
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']
simple_ci = model_simple.conf_int().loc['beauty'].tolist()

ctrl_coef = model_controls.params['beauty']
ctrl_p = model_controls.pvalues['beauty']
ctrl_ci = model_controls.conf_int().loc['beauty'].tolist()

# Standardize effect size: per 1 SD beauty -> change in eval
beauty_sd = _df_key['beauty'].std()
# Using simple model slope as primary for effect size
std_effect = simple_coef * beauty_sd

# Prepare summary dict for writing to console (debug) only
summary = {
    'n_simple': int(model_simple.nobs),
    'n_controls': int(model_controls.nobs),
    'corr_eval_beauty': corr,
    'simple_coef': simple_coef,
    'simple_p': simple_p,
    'simple_ci': simple_ci,
    'ctrl_coef': ctrl_coef,
    'ctrl_p': ctrl_p,
    'ctrl_ci': ctrl_ci,
    'beauty_sd': beauty_sd,
    'std_effect_eval': std_effect,
}

print(json.dumps(summary, indent=2))
