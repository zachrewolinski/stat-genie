import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived variables
df['size_diff'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']  # positive means contest closer to focal center
df['loc_ratio'] = df['dist_other'] / df['dist_focal']

# Prepare response
y = df['win']

def fit_logit(predictors):
    X = df[predictors].copy()
    X = sm.add_constant(X, has_constant='add')
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit()
    # robust SE (HC3) via refit for compatibility
    res_robust = model.fit(cov_type='HC3')
    return res, res_robust

models = {}
for name, preds in [
    ('size_only', ['size_diff']),
    ('loc_only', ['loc_adv']),
    ('size_loc', ['size_diff', 'loc_adv']),
    ('size_ratio_loc', ['size_ratio', 'loc_adv']),
]:
    res, res_robust = fit_logit(preds)
    models[name] = (preds, res, res_robust)

summary = {}
for name, (preds, res, res_robust) in models.items():
    # odds ratios and CI from robust results
    params = res_robust.params
    conf = res_robust.conf_int()
    or_params = np.exp(params)
    or_conf = np.exp(conf)
    pvals = res_robust.pvalues
    summary[name] = {
        'n': int(len(df)),
        'preds': preds,
        'params': params.to_dict(),
        'pvalues': pvals.to_dict(),
        'odds_ratio': or_params.to_dict(),
        'or_ci_low': or_conf[0].to_dict(),
        'or_ci_high': or_conf[1].to_dict(),
        'aic': float(res.aic),
        'llf': float(res.llf),
    }

# Simple descriptive checks
win_rate = df['win'].mean()

# Grouped win rates by sign of size_diff and loc_adv
size_groups = df.groupby(df['size_diff'] > 0)['win'].mean().to_dict()
loc_groups = df.groupby(df['loc_adv'] > 0)['win'].mean().to_dict()

out = {
    'win_rate': float(win_rate),
    'size_groups_win_rate': {str(k): float(v) for k, v in size_groups.items()},
    'loc_groups_win_rate': {str(k): float(v) for k, v in loc_groups.items()},
    'models': summary,
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
