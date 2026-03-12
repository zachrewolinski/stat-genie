import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
DF = pd.read_csv('teachingratings.csv')

# Basic clean
# Ensure expected columns

# Simple correlation
corr = DF['beauty'].corr(DF['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=DF).fit()

# Multiple regression with controls
# Use categorical variables via C()
# Avoid perfect multicollinearity: students and allstudents highly correlated; keep students only
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students'
model_ctrl = smf.ols(formula, data=DF).fit()

# Robust regression with HC3 to handle heteroskedasticity
model_ctrl_robust = model_ctrl.get_robustcov_results(cov_type='HC3')

# Also check standardized effect size (beta) for beauty in controlled model
# standardize numeric variables
num_cols = ['eval','beauty','age','students']
DF_std = DF.copy()
for c in num_cols:
    DF_std[c] = (DF_std[c] - DF_std[c].mean()) / DF_std[c].std(ddof=0)

model_ctrl_std = smf.ols('eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students', data=DF_std).fit()
model_ctrl_std_robust = model_ctrl_std.get_robustcov_results(cov_type='HC3')

# Output summary stats we need
results = {
    'n': int(model_simple.nobs),
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'ctrl_coef': model_ctrl.params['beauty'],
    'ctrl_p': model_ctrl.pvalues['beauty'],
    'ctrl_ci': model_ctrl.conf_int().loc['beauty'].tolist(),
    'ctrl_robust_p': model_ctrl_robust.pvalues[model_ctrl_robust.model.exog_names.index('beauty')],
    'ctrl_robust_ci': model_ctrl_robust.conf_int()[model_ctrl_robust.model.exog_names.index('beauty')].tolist(),
    'ctrl_r2': model_ctrl.rsquared,
    'ctrl_std_coef': model_ctrl_std.params['beauty'],
    'ctrl_std_p': model_ctrl_std.pvalues['beauty'],
    'ctrl_std_robust_p': model_ctrl_std_robust.pvalues[model_ctrl_std_robust.model.exog_names.index('beauty')],
}

print(json.dumps(results, indent=2))
