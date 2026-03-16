import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('crofoot.csv')

# Variables per metadata
outcome = df['m_focal']  # 1 if focal won
size_focal = df['f_other']
size_other = df['win']
size_diff = size_focal - size_other
loc_adv = df['n_focal'] - df['m_other']  # positive => focal closer to its center than other

# Logistic regression with size_diff and location advantage
X = pd.DataFrame({
    'size_diff': size_diff,
    'loc_adv': loc_adv,
})
X = sm.add_constant(X)
model = sm.Logit(outcome, X)
res = model.fit(disp=False)

# Odds ratios and p-values
coef = res.params
pvals = res.pvalues
or_vals = np.exp(coef)

# Predicted probability change from 25th to 75th percentile for each predictor
med_size_diff = float(np.median(size_diff))
med_loc_adv = float(np.median(loc_adv))

q25_size, q75_size = np.percentile(size_diff, [25, 75])
q25_loc, q75_loc = np.percentile(loc_adv, [25, 75])

# helper to compute predicted prob
from scipy.special import expit

def pred_prob(size_d, loc_d):
    xb = coef['const'] + coef['size_diff'] * size_d + coef['loc_adv'] * loc_d
    return float(expit(xb))

prob_size_low = pred_prob(q25_size, med_loc_adv)
prob_size_high = pred_prob(q75_size, med_loc_adv)
prob_loc_low = pred_prob(med_size_diff, q25_loc)
prob_loc_high = pred_prob(med_size_diff, q75_loc)

# t-tests for group differences
size_t, size_p = stats.ttest_ind(size_diff[outcome==1], size_diff[outcome==0], equal_var=False)
loc_t, loc_p = stats.ttest_ind(loc_adv[outcome==1], loc_adv[outcome==0], equal_var=False)

results = {
    'n': int(len(df)),
    'wins': int(outcome.sum()),
    'losses': int((1-outcome).sum()),
    'coef': coef.to_dict(),
    'pvals': pvals.to_dict(),
    'odds_ratios': or_vals.to_dict(),
    'prob_change': {
        'size_diff_q25': float(q25_size),
        'size_diff_q75': float(q75_size),
        'prob_size_low': prob_size_low,
        'prob_size_high': prob_size_high,
        'loc_adv_q25': float(q25_loc),
        'loc_adv_q75': float(q75_loc),
        'prob_loc_low': prob_loc_low,
        'prob_loc_high': prob_loc_high,
    },
    'ttests': {
        'size_diff_t': float(size_t),
        'size_diff_p': float(size_p),
        'loc_adv_t': float(loc_t),
        'loc_adv_p': float(loc_p),
    }
}

import json
print(json.dumps(results, indent=2))
