import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Create relative size and contest location variables
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_size_ratio'] = _df['n_focal'] / _df['n_other']

# Location: relative distance to home range center (positive = closer to focal than other)
_df['rel_dist'] = _df['dist_other'] - _df['dist_focal']
# Alternative: ratio
_df['rel_dist_ratio'] = _df['dist_other'] / _df['dist_focal']

# Standardize predictors for interpretability
for col in ['rel_size', 'rel_size_ratio', 'rel_dist', 'rel_dist_ratio']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Logistic regression with difference metrics
model_diff = smf.logit('win ~ rel_size_z + rel_dist_z', data=_df).fit(disp=0)

# Logistic regression with ratio metrics
model_ratio = smf.logit('win ~ rel_size_ratio_z + rel_dist_ratio_z', data=_df).fit(disp=0)

# Also check models separately
model_size_only = smf.logit('win ~ rel_size_z', data=_df).fit(disp=0)
model_loc_only = smf.logit('win ~ rel_dist_z', data=_df).fit(disp=0)

print('N:', len(_df))
print('\nModel (diff) summary:')
print(model_diff.summary())
print('\nModel (ratio) summary:')
print(model_ratio.summary())
print('\nSize only:')
print(model_size_only.summary())
print('\nLocation only:')
print(model_loc_only.summary())

# Calculate marginal effects for diff model
margeff = model_diff.get_margeff(at='mean')
print('\nMarginal effects (diff model at mean):')
print(margeff.summary())

# Save key statistics to a json-like dict for later
results = {
    'diff_params': model_diff.params.to_dict(),
    'diff_pvalues': model_diff.pvalues.to_dict(),
    'ratio_params': model_ratio.params.to_dict(),
    'ratio_pvalues': model_ratio.pvalues.to_dict(),
    'size_only_p': model_size_only.pvalues.to_dict(),
    'loc_only_p': model_loc_only.pvalues.to_dict(),
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
