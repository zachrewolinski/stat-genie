import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Feature engineering
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['dist_diff'] = _df['dist_other'] - _df['dist_focal']
_df['dist_sum'] = _df['dist_other'] + _df['dist_focal']

# Logistic regression: win ~ size_diff + dist_diff
X = _df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
y = _df['win']

model = sm.Logit(y, X)
res = model.fit(disp=False)

# Alternative model: win ~ size_ratio + dist_diff (just in case)
X2 = sm.add_constant(_df[['size_ratio', 'dist_diff']])
res2 = sm.Logit(y, X2).fit(disp=False)

# Simple bivariate checks (t-tests) for intuition
# Compare size_diff and dist_diff by win
win_groups = _df.groupby('win')
mean_size_diff = win_groups['size_diff'].mean().to_dict()
mean_dist_diff = win_groups['dist_diff'].mean().to_dict()

# Export results to JSON for parsing
output = {
    'n_rows': int(len(_df)),
    'logit_coef': res.params.to_dict(),
    'logit_pvalues': res.pvalues.to_dict(),
    'logit_confint': {k: list(v) for k, v in res.conf_int().to_dict(orient='index').items()},
    'logit_llf': float(res.llf),
    'logit_aic': float(res.aic),
    'logit2_coef': res2.params.to_dict(),
    'logit2_pvalues': res2.pvalues.to_dict(),
    'mean_size_diff_by_win': mean_size_diff,
    'mean_dist_diff_by_win': mean_dist_diff,
    'size_diff_sd': float(_df['size_diff'].std()),
    'dist_diff_sd': float(_df['dist_diff'].std()),
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
