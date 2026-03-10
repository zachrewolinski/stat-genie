import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning: drop rows with missing eval or beauty
_df = _df.dropna(subset=['eval', 'beauty']).copy()

# Descriptives
n = len(_df)
eval_mean = _df['eval'].mean()
eval_sd = _df['eval'].std(ddof=1)
beauty_mean = _df['beauty'].mean()
beauty_sd = _df['beauty'].std(ddof=1)

# Pearson correlation
r, p_r = stats.pearsonr(_df['beauty'], _df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC3')

# Controls model
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
)
model_controls = smf.ols(formula, data=_df).fit(cov_type='HC3')

# Standardized effect (beauty in SD units)
_df['beauty_z'] = ( _df['beauty'] - beauty_mean) / beauty_sd
model_simple_z = smf.ols('eval ~ beauty_z', data=_df).fit(cov_type='HC3')
model_controls_z = smf.ols(
    'eval ~ beauty_z + age + students + allstudents '
    '+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)',
    data=_df
).fit(cov_type='HC3')

# Collect key stats
results = {
    'n': n,
    'eval_mean': eval_mean,
    'eval_sd': eval_sd,
    'beauty_mean': beauty_mean,
    'beauty_sd': beauty_sd,
    'pearson_r': r,
    'pearson_p': p_r,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'simple_coef_sd_units': model_simple_z.params['beauty_z'],
    'simple_p_sd_units': model_simple_z.pvalues['beauty_z'],
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_r2': model_controls.rsquared,
    'controls_coef_sd_units': model_controls_z.params['beauty_z'],
    'controls_p_sd_units': model_controls_z.pvalues['beauty_z'],
}

print(json.dumps(results, indent=2))
