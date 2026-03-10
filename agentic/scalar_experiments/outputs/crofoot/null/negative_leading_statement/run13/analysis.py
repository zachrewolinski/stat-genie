import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Derived variables
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_size_ratio'] = _df['n_focal'] / _df['n_other']
# Positive when focal is closer to its own home-range center than the other group is to its center
_df['rel_location'] = _df['dist_other'] - _df['dist_focal']

# Standardize predictors for comparability
for col in ['rel_size', 'rel_size_ratio', 'rel_location']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Logistic regressions
# Model with rel_size and rel_location
model1 = smf.logit('win ~ rel_size_z + rel_location_z', data=_df).fit(disp=False)

# Alternative model with size ratio
model2 = smf.logit('win ~ rel_size_ratio_z + rel_location_z', data=_df).fit(disp=False)

# Individual predictor models
model_size = smf.logit('win ~ rel_size_z', data=_df).fit(disp=False)
model_loc = smf.logit('win ~ rel_location_z', data=_df).fit(disp=False)

# Extract summaries
summary = {
    'n_rows': len(_df),
    'win_rate': _df['win'].mean(),
    'rel_size_mean': _df['rel_size'].mean(),
    'rel_location_mean': _df['rel_location'].mean(),
    'model1_params': model1.params.to_dict(),
    'model1_pvalues': model1.pvalues.to_dict(),
    'model1_conf_int': model1.conf_int().to_dict(),
    'model2_params': model2.params.to_dict(),
    'model2_pvalues': model2.pvalues.to_dict(),
    'model2_conf_int': model2.conf_int().to_dict(),
    'model_size_params': model_size.params.to_dict(),
    'model_size_pvalues': model_size.pvalues.to_dict(),
    'model_loc_params': model_loc.params.to_dict(),
    'model_loc_pvalues': model_loc.pvalues.to_dict(),
}

# Save summary to inspect
import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
