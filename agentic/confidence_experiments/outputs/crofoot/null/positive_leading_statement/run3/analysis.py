import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('crofoot.csv')

# Create predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']  # positive => focal closer to its center

# Basic stats
n = len(_df)
win_rate = _df['win'].mean()

# Logistic regression with size_diff and loc_adv
X = _df[['size_diff', 'loc_adv']]
X = sm.add_constant(X)
model = sm.Logit(_df['win'], X)
res = model.fit(disp=False)

# Alternative with size_ratio (log) and loc_adv for robustness
_df['log_size_ratio'] = np.log(_df['size_ratio'])
X2 = _df[['log_size_ratio', 'loc_adv']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(_df['win'], X2)
res2 = model2.fit(disp=False)

# Simple group comparisons
size_adv = _df['size_diff'] > 0
loc_adv = _df['loc_adv'] > 0

# Win rates by advantages
win_size_adv = _df.loc[size_adv, 'win'].mean()
win_size_disadv = _df.loc[~size_adv, 'win'].mean()
win_loc_adv = _df.loc[loc_adv, 'win'].mean()
win_loc_disadv = _df.loc[~loc_adv, 'win'].mean()

# t-tests for mean differences in predictors by win
size_diff_win = _df.loc[_df['win'] == 1, 'size_diff']
size_diff_loss = _df.loc[_df['win'] == 0, 'size_diff']
loc_adv_win = _df.loc[_df['win'] == 1, 'loc_adv']
loc_adv_loss = _df.loc[_df['win'] == 0, 'loc_adv']

t_size = stats.ttest_ind(size_diff_win, size_diff_loss, equal_var=False)
t_loc = stats.ttest_ind(loc_adv_win, loc_adv_loss, equal_var=False)

# Odds ratios from logistic regression
params = res.params
conf = res.conf_int()
OR = np.exp(params)
OR_ci = np.exp(conf)

params2 = res2.params
conf2 = res2.conf_int()
OR2 = np.exp(params2)
OR2_ci = np.exp(conf2)

# Collect outputs
out = {
    'n': int(n),
    'win_rate': float(win_rate),
    'logit_size_diff_loc_adv': {
        'params': res.params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'odds_ratio': OR.to_dict(),
        'odds_ratio_ci': {k: [float(OR_ci.loc[k,0]), float(OR_ci.loc[k,1])] for k in OR_ci.index},
        'pseudo_r2': float(res.prsquared),
        'llf': float(res.llf),
    },
    'logit_log_size_ratio_loc_adv': {
        'params': res2.params.to_dict(),
        'pvalues': res2.pvalues.to_dict(),
        'odds_ratio': OR2.to_dict(),
        'odds_ratio_ci': {k: [float(OR2_ci.loc[k,0]), float(OR2_ci.loc[k,1])] for k in OR2_ci.index},
        'pseudo_r2': float(res2.prsquared),
        'llf': float(res2.llf),
    },
    'win_rates': {
        'size_adv': float(win_size_adv),
        'size_disadv': float(win_size_disadv),
        'loc_adv': float(win_loc_adv),
        'loc_disadv': float(win_loc_disadv),
    },
    'ttests': {
        'size_diff': {'t': float(t_size.statistic), 'p': float(t_size.pvalue)},
        'loc_adv': {'t': float(t_loc.statistic), 'p': float(t_loc.pvalue)},
    },
}

print(json.dumps(out, indent=2))
