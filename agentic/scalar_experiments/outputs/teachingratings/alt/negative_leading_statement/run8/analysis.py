import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure numeric types
_df['beauty'] = pd.to_numeric(_df['beauty'], errors='coerce')
_df['eval'] = pd.to_numeric(_df['eval'], errors='coerce')

# Drop missing
_df = _df.dropna(subset=['beauty','eval'])

n = len(_df)

# Correlation
corr, corr_p = stats.pearsonr(_df['beauty'], _df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit()

# Controlled OLS with available covariates
# Use categorical variables as C(); numeric: age, students, allstudents
# Exclude rownames, prof as identifiers
formula = 'eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)'
model_ctrl = smf.ols(formula, data=_df).fit()

# Collect key stats
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']
simple_ci = model_simple.conf_int().loc['beauty'].tolist()

ctrl_coef = model_ctrl.params['beauty']
ctrl_p = model_ctrl.pvalues['beauty']
ctrl_ci = model_ctrl.conf_int().loc['beauty'].tolist()

# Effect size in SDs: beauty is standardized? mean zero but not SD=1. We'll compute effect per SD of beauty.
beauty_sd = _df['beauty'].std()
eval_sd = _df['eval'].std()
# predicted eval change for 1 SD increase in beauty
simple_effect_sd = simple_coef * beauty_sd
ctrl_effect_sd = ctrl_coef * beauty_sd

# R-squared for simple
r2_simple = model_simple.rsquared
r2_ctrl = model_ctrl.rsquared

# Save summary stats for reporting
summary = {
    'n': n,
    'corr': corr,
    'corr_p': corr_p,
    'simple_coef': simple_coef,
    'simple_p': simple_p,
    'simple_ci_low': simple_ci[0],
    'simple_ci_high': simple_ci[1],
    'ctrl_coef': ctrl_coef,
    'ctrl_p': ctrl_p,
    'ctrl_ci_low': ctrl_ci[0],
    'ctrl_ci_high': ctrl_ci[1],
    'beauty_sd': beauty_sd,
    'eval_sd': eval_sd,
    'simple_effect_sd': simple_effect_sd,
    'ctrl_effect_sd': ctrl_effect_sd,
    'r2_simple': r2_simple,
    'r2_ctrl': r2_ctrl
}

print(summary)
