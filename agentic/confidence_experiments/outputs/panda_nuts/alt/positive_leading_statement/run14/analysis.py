import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Clean / compute efficiency: nuts opened per second
# Avoid division by zero
if (df['seconds'] <= 0).any():
    raise ValueError('Non-positive seconds encountered')

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Encode categorical variables for modeling
# Use C() in formula to handle categorical

# Basic linear regression
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()

# Also consider log efficiency if skewed (add small constant)
# Not primary, but check robustness

df['log_eff'] = np.log(df['efficiency'] + 1e-6)
model_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit()

# Output summaries
print('N=', len(df))
print('Efficiency summary:')
print(df['efficiency'].describe())
print('\nOLS model:')
print(model.summary())
print('\nOLS log-eff model:')
print(model_log.summary())

# Save key results for quick parsing
results = {
    'n': len(df),
    'eff_mean': df['efficiency'].mean(),
    'eff_sd': df['efficiency'].std(),
    'eff_min': df['efficiency'].min(),
    'eff_max': df['efficiency'].max(),
    'ols_params': model.params.to_dict(),
    'ols_pvalues': model.pvalues.to_dict(),
    'ols_r2': model.rsquared,
    'ols_adj_r2': model.rsquared_adj,
    'log_params': model_log.params.to_dict(),
    'log_pvalues': model_log.pvalues.to_dict(),
    'log_r2': model_log.rsquared,
    'log_adj_r2': model_log.rsquared_adj,
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
