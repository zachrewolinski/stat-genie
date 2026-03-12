import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Engineer predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
# Positive means focal is closer to its own center than the other group is to its own center
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']

# Helper to fit logit model safely

def fit_logit(y, X):
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    res = model.fit(disp=False)
    return res

results = {}

y = _df['win']

# Primary model: relative size + relative location
res_main = fit_logit(y, _df[['size_diff', 'loc_adv']])
results['main'] = {
    'params': {k: float(v) for k, v in res_main.params.items()},
    'pvalues': {k: float(v) for k, v in res_main.pvalues.items()},
    'aic': float(res_main.aic),
}

# Standardized version for effect size
X_std = (_df[['size_diff', 'loc_adv']] - _df[['size_diff', 'loc_adv']].mean()) / _df[['size_diff', 'loc_adv']].std(ddof=0)
res_main_std = fit_logit(y, X_std)
results['main_std'] = {
    'params': {k: float(v) for k, v in res_main_std.params.items()},
    'pvalues': {k: float(v) for k, v in res_main_std.pvalues.items()},
}

# Alternative: size ratio + relative location
res_ratio = fit_logit(y, _df[['size_ratio', 'loc_adv']])
results['ratio'] = {
    'params': {k: float(v) for k, v in res_ratio.params.items()},
    'pvalues': {k: float(v) for k, v in res_ratio.pvalues.items()},
    'aic': float(res_ratio.aic),
}

# Alternative: absolute distances
res_abs = fit_logit(y, _df[['size_diff', 'dist_focal', 'dist_other']])
results['absolute'] = {
    'params': {k: float(v) for k, v in res_abs.params.items()},
    'pvalues': {k: float(v) for k, v in res_abs.pvalues.items()},
    'aic': float(res_abs.aic),
}

# Single predictor models
res_size_only = fit_logit(y, _df[['size_diff']])
res_loc_only = fit_logit(y, _df[['loc_adv']])
results['size_only'] = {
    'params': {k: float(v) for k, v in res_size_only.params.items()},
    'pvalues': {k: float(v) for k, v in res_size_only.pvalues.items()},
    'aic': float(res_size_only.aic),
}
results['loc_only'] = {
    'params': {k: float(v) for k, v in res_loc_only.params.items()},
    'pvalues': {k: float(v) for k, v in res_loc_only.pvalues.items()},
    'aic': float(res_loc_only.aic),
}

# Predicted probability change for +1 SD in each predictor (holding other at mean) using main model
means = _df[['size_diff', 'loc_adv']].mean()
stds = _df[['size_diff', 'loc_adv']].std(ddof=0)
base = pd.DataFrame({'const':[1.0], 'size_diff':[means['size_diff']], 'loc_adv':[means['loc_adv']]})
inc_size = pd.DataFrame({'const':[1.0], 'size_diff':[means['size_diff'] + stds['size_diff']], 'loc_adv':[means['loc_adv']]})
inc_loc = pd.DataFrame({'const':[1.0], 'size_diff':[means['size_diff']], 'loc_adv':[means['loc_adv'] + stds['loc_adv']]})

p_base = float(res_main.predict(base))
p_size = float(res_main.predict(inc_size))
p_loc = float(res_main.predict(inc_loc))

# Descriptive win rates
win_rate_size_adv = {
    'size_diff>0': float(_df.loc[_df['size_diff'] > 0, 'win'].mean()),
    'size_diff=0': float(_df.loc[_df['size_diff'] == 0, 'win'].mean()) if (_df['size_diff'] == 0).any() else None,
    'size_diff<0': float(_df.loc[_df['size_diff'] < 0, 'win'].mean()),
}

win_rate_loc_adv = {
    'loc_adv>0': float(_df.loc[_df['loc_adv'] > 0, 'win'].mean()),
    'loc_adv=0': float(_df.loc[_df['loc_adv'] == 0, 'win'].mean()) if (_df['loc_adv'] == 0).any() else None,
    'loc_adv<0': float(_df.loc[_df['loc_adv'] < 0, 'win'].mean()),
}

out = {
    'n': int(len(_df)),
    'results': results,
    'p_base': p_base,
    'p_size_plus_1sd': p_size,
    'p_loc_plus_1sd': p_loc,
    'win_rate_size_adv': win_rate_size_adv,
    'win_rate_loc_adv': win_rate_loc_adv,
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
