import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic stats
n = len(_df)

# Simple correlation
corr = _df['beauty'].corr(_df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC3')

# With controls (as per dataset typical controls)
# Treat categorical vars as categories
cat_vars = ['minority','gender','credits','division','native','tenure']
for c in cat_vars:
    _df[c] = _df[c].astype('category')

# Use students (participating) and allstudents as continuous controls
# include age
formula = 'eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)'
model_ctrl = smf.ols(formula, data=_df).fit(cov_type='HC3')

# Effect size per SD of beauty
beauty_sd = _df['beauty'].std()
coef_simple = model_simple.params['beauty']
coef_ctrl = model_ctrl.params['beauty']

# Predicted change for 1 SD increase
sd_effect_simple = coef_simple * beauty_sd
sd_effect_ctrl = coef_ctrl * beauty_sd

# Save key results
results = {
    'n': n,
    'corr': corr,
    'simple_coef': coef_simple,
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'ctrl_coef': coef_ctrl,
    'ctrl_p': model_ctrl.pvalues['beauty'],
    'ctrl_ci': model_ctrl.conf_int().loc['beauty'].tolist(),
    'beauty_sd': beauty_sd,
    'sd_effect_simple': sd_effect_simple,
    'sd_effect_ctrl': sd_effect_ctrl,
    'r2_simple': model_simple.rsquared,
    'r2_ctrl': model_ctrl.rsquared,
}

import json
print(json.dumps(results, indent=2))
