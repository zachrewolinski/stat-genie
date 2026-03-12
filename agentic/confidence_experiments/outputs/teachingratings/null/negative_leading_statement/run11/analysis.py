import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic clean: drop rows with missing eval or beauty
_df = _df.dropna(subset=['eval', 'beauty']).copy()

# Bivariate correlation
corr = _df['beauty'].corr(_df['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC1')

# Add controls (based on available fields)
# Encode categorical variables with C() for statsmodels
formula_controls = (
    'eval ~ beauty + age + students + allstudents + C(gender) + C(minority) '
    '+ C(credits) + C(division) + C(native) + C(tenure)'
)
model_controls = smf.ols(formula_controls, data=_df).fit(cov_type='HC1')

# Also include professor fixed effects? maybe too many; but can check robust
# We'll skip fixed effects in primary due to small sample per prof.

results = {
    'n': int(model_simple.nobs),
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'r2_simple': model_simple.rsquared,
    'r2_controls': model_controls.rsquared,
}

print(json.dumps(results, indent=2))
