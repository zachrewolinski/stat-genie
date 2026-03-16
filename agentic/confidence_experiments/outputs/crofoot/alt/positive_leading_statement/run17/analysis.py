import json
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
_df = pd.read_csv('crofoot.csv')

# Feature engineering
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['location_adv'] = _df['dist_other'] - _df['dist_focal']  # positive => closer to focal home range center

# Standardize for effect size comparisons
for col in ['size_diff', 'location_adv']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Base model: size_diff + location_adv
X = sm.add_constant(_df[['size_diff', 'location_adv']])
model = sm.GLM(_df['win'], X, family=sm.families.Binomial())
res = model.fit()

# Standardized model for comparable effect sizes
Xz = sm.add_constant(_df[['size_diff_z', 'location_adv_z']])
model_z = sm.GLM(_df['win'], Xz, family=sm.families.Binomial())
res_z = model_z.fit()

# Alternative model using size_ratio to check robustness
Xr = sm.add_constant(_df[['size_ratio', 'location_adv']])
model_r = sm.GLM(_df['win'], Xr, family=sm.families.Binomial())
res_r = model_r.fit()

# Compute predicted probability difference for 1 SD change in each predictor using standardized model
coef_z = res_z.params
# baseline at 0 (mean) -> logit = const; then add 1 SD
baseline_logit = coef_z['const']

def logistic(x):
    return 1 / (1 + np.exp(-x))

pred_base = logistic(baseline_logit)
pred_size = logistic(baseline_logit + coef_z['size_diff_z'])
pred_loc = logistic(baseline_logit + coef_z['location_adv_z'])

out = {
    'n': len(_df),
    'summary_main': res.summary2().as_text(),
    'summary_z': res_z.summary2().as_text(),
    'summary_ratio': res_r.summary2().as_text(),
    'pred_base': float(pred_base),
    'pred_size_1sd': float(pred_size),
    'pred_loc_1sd': float(pred_loc),
    'coef_main': res.params.to_dict(),
    'pvalues_main': res.pvalues.to_dict(),
    'coef_z': res_z.params.to_dict(),
    'pvalues_z': res_z.pvalues.to_dict(),
    'coef_ratio': res_r.params.to_dict(),
    'pvalues_ratio': res_r.pvalues.to_dict(),
}

print(json.dumps(out, indent=2))
