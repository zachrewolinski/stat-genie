import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Outcome: nut-cracking efficiency (nuts per second)
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Encode categorical predictors
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# Fit linear model for efficiency
# Using robust (HC3) standard errors for heteroskedasticity
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')

# Also fit a log-efficiency model as a sensitivity check
_df['log_efficiency'] = np.log(_df['efficiency'])
log_model = smf.ols('log_efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')

# Collect key results
summary = {
    'n': int(len(_df)),
    'efficiency_mean': float(_df['efficiency'].mean()),
    'efficiency_std': float(_df['efficiency'].std()),
    'ols_params': model.params.to_dict(),
    'ols_pvalues': model.pvalues.to_dict(),
    'ols_rsquared': float(model.rsquared),
    'log_params': log_model.params.to_dict(),
    'log_pvalues': log_model.pvalues.to_dict(),
    'log_rsquared': float(log_model.rsquared),
}

print('Linear model (efficiency) with robust SE')
print(model.summary())
print('\nLog-efficiency model (robust SE)')
print(log_model.summary())
print('\nSummary dict')
print(summary)
