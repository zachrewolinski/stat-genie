import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

# Load data
_df = pd.read_csv('crofoot.csv')

# Create predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['dist_diff'] = _df['dist_focal'] - _df['dist_other']  # positive means focal farther from its center

# Basic summaries
n = len(_df)
win_rate = _df['win'].mean()

# Logistic regression
X = _df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
model = sm.Logit(_df['win'], X)
res = model.fit(disp=False)

# Also compute standardized effects for interpretability
X_std = _df[['size_diff', 'dist_diff']].copy()
X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
X_std = sm.add_constant(X_std)
res_std = sm.Logit(_df['win'], X_std).fit(disp=False)

# Odds ratios
params = res.params
conf = res.conf_int()
odds_ratios = np.exp(params)
conf_or = np.exp(conf)

# Predictive check: effect of size_diff and dist_diff
# For size_diff, compare win prob at -1, 0, +1 (one monkey difference)
# For dist_diff, compare -100, 0, +100 meters

def predict_prob(size_diff, dist_diff):
    x = np.array([1.0, size_diff, dist_diff])
    return float(res.predict(x))

preds = {
    'size_minus1': predict_prob(-1, 0),
    'size_0': predict_prob(0, 0),
    'size_plus1': predict_prob(1, 0),
    'dist_minus100': predict_prob(0, -100),
    'dist_0': predict_prob(0, 0),
    'dist_plus100': predict_prob(0, 100),
}

output = {
    'n': n,
    'win_rate': win_rate,
    'logit_params': res.params.to_dict(),
    'logit_pvalues': res.pvalues.to_dict(),
    'logit_conf_int': {k: [float(conf.loc[k,0]), float(conf.loc[k,1])] for k in conf.index},
    'odds_ratios': odds_ratios.to_dict(),
    'odds_ratio_conf_int': {k: [float(conf_or.loc[k,0]), float(conf_or.loc[k,1])] for k in conf_or.index},
    'logit_params_std': res_std.params.to_dict(),
    'logit_pvalues_std': res_std.pvalues.to_dict(),
    'preds': preds,
}

print(json.dumps(output, indent=2))
