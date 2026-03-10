import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleanliness: ensure no missing in key vars
# compute simple correlation
corr = _df['beauty'].corr(_df['eval'])

# Simple regression: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=_df).fit()

# Multivariate regression with common controls from dataset
# Use categorical variables with C()
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
)
model_ctrl = smf.ols(formula, data=_df).fit()

# Collect results
results = {
    'n': int(model_simple.nobs),
    'corr': corr,
    'simple': {
        'coef': model_simple.params['beauty'],
        'se': model_simple.bse['beauty'],
        't': model_simple.tvalues['beauty'],
        'p': model_simple.pvalues['beauty'],
        'r2': model_simple.rsquared,
    },
    'ctrl': {
        'coef': model_ctrl.params['beauty'],
        'se': model_ctrl.bse['beauty'],
        't': model_ctrl.tvalues['beauty'],
        'p': model_ctrl.pvalues['beauty'],
        'r2': model_ctrl.rsquared,
    }
}

# Pretty print for inspection
import json
print(json.dumps(results, indent=2))
