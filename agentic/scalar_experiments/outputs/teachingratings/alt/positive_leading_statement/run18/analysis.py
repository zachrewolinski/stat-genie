import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic correlation
r, p = stats.pearsonr(df['beauty'], df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multivariate OLS with common controls
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
)
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

beauty_std = df['beauty'].std()

results = {
    'n': int(df.shape[0]),
    'corr_r': float(r),
    'corr_p': float(p),
    'simple_coef': float(model_simple.params['beauty']),
    'simple_p': float(model_simple.pvalues['beauty']),
    'simple_ci': [float(x) for x in model_simple.conf_int().loc['beauty']],
    'controls_coef': float(model_controls.params['beauty']),
    'controls_p': float(model_controls.pvalues['beauty']),
    'controls_ci': [float(x) for x in model_controls.conf_int().loc['beauty']],
    'beauty_sd': float(beauty_std),
    'simple_effect_1sd': float(model_simple.params['beauty'] * beauty_std),
    'controls_effect_1sd': float(model_controls.params['beauty'] * beauty_std),
    'simple_r2': float(model_simple.rsquared),
    'controls_r2': float(model_controls.rsquared),
}

# Print results for review
for k, v in results.items():
    print(f"{k}: {v}")
