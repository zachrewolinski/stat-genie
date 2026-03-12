import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Outcome
_df['win'] = _df['feature4']

# Predictors
_df['size_diff'] = _df['feature7'] - _df['feature8']
_df['size_ratio'] = _df['feature7'] / _df['feature8']
_df['loc_diff'] = _df['feature5'] - _df['feature6']
_df['loc_ratio'] = _df['feature5'] / _df['feature6']

# Standardize continuous predictors for interpretability
for col in ['size_diff', 'size_ratio', 'loc_diff', 'loc_ratio']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

results = {}

# Main model: size_diff and loc_diff
model1 = smf.logit('win ~ size_diff_z + loc_diff_z', data=_df).fit(disp=False)
results['model1'] = model1.summary2().tables[1]

# Alternative model: size_ratio and loc_diff
model2 = smf.logit('win ~ size_ratio_z + loc_diff_z', data=_df).fit(disp=False)
results['model2'] = model2.summary2().tables[1]

# Alternative model: size_diff and loc_ratio
model3 = smf.logit('win ~ size_diff_z + loc_ratio_z', data=_df).fit(disp=False)
results['model3'] = model3.summary2().tables[1]

# Simple bivariate checks (for directionality)
# Use logistic regression with single predictors
model4 = smf.logit('win ~ size_diff_z', data=_df).fit(disp=False)
model5 = smf.logit('win ~ loc_diff_z', data=_df).fit(disp=False)
results['model4'] = model4.summary2().tables[1]
results['model5'] = model5.summary2().tables[1]

# Collect key stats
out = {
    'n': int(_df.shape[0]),
    'win_rate': float(_df['win'].mean()),
    'models': {
        'size_diff_loc_diff': {
            'params': model1.params.to_dict(),
            'pvalues': model1.pvalues.to_dict(),
            'conf_int': model1.conf_int().to_dict(),
            'pseudo_r2': float(model1.prsquared),
        },
        'size_ratio_loc_diff': {
            'params': model2.params.to_dict(),
            'pvalues': model2.pvalues.to_dict(),
            'conf_int': model2.conf_int().to_dict(),
            'pseudo_r2': float(model2.prsquared),
        },
        'size_diff_loc_ratio': {
            'params': model3.params.to_dict(),
            'pvalues': model3.pvalues.to_dict(),
            'conf_int': model3.conf_int().to_dict(),
            'pseudo_r2': float(model3.prsquared),
        },
        'size_diff_only': {
            'params': model4.params.to_dict(),
            'pvalues': model4.pvalues.to_dict(),
            'conf_int': model4.conf_int().to_dict(),
            'pseudo_r2': float(model4.prsquared),
        },
        'loc_diff_only': {
            'params': model5.params.to_dict(),
            'pvalues': model5.pvalues.to_dict(),
            'conf_int': model5.conf_int().to_dict(),
            'pseudo_r2': float(model5.prsquared),
        },
    },
}

print(json.dumps(out, indent=2))
