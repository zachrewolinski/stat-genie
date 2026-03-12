import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Map columns based on info.json descriptions
# Outcome: 1 if focal won contest, 0 if other won
outcome_col = 'm_focal'

# Distances from home range centers
# m_other: distance of focal group from center of its home range
# n_focal: distance of other group from center of its home range
focal_dist_col = 'm_other'
other_dist_col = 'n_focal'

# Group sizes
# f_other: number of individuals in focal group
# win: number of individuals in other group
focal_size_col = 'f_other'
other_size_col = 'win'

# Compute relative predictors
# Relative size > 0 => focal larger
_df['rel_size'] = _df[focal_size_col] - _df[other_size_col]

# Relative location > 0 => contest closer to focal center (other distance greater)
_df['rel_location'] = _df[other_dist_col] - _df[focal_dist_col]

# Basic checks
n = len(_df)
win_rate = _df[outcome_col].mean()

# Logistic regression with both predictors
X = _df[['rel_size', 'rel_location']].copy()
X = sm.add_constant(X)
model = sm.Logit(_df[outcome_col], X)
res = model.fit(disp=False)

# Also fit single-predictor models for robustness
X_size = sm.add_constant(_df[['rel_size']])
res_size = sm.Logit(_df[outcome_col], X_size).fit(disp=False)

X_loc = sm.add_constant(_df[['rel_location']])
res_loc = sm.Logit(_df[outcome_col], X_loc).fit(disp=False)

# Compute odds ratios and CIs
params = res.params
conf = res.conf_int()
conf.columns = ['ci_low', 'ci_high']

odds = np.exp(params)
conf_odds = np.exp(conf)

# Standardize predictors for comparable effect sizes
X_std = _df[['rel_size', 'rel_location']].copy()
X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
X_std = sm.add_constant(X_std)
res_std = sm.Logit(_df[outcome_col], X_std).fit(disp=False)

summary = {
    'n': int(n),
    'win_rate': float(win_rate),
    'coef': res.params.to_dict(),
    'pvalues': res.pvalues.to_dict(),
    'odds_ratio': odds.to_dict(),
    'odds_ci_low': conf_odds['ci_low'].to_dict(),
    'odds_ci_high': conf_odds['ci_high'].to_dict(),
    'single_pvalues': {
        'rel_size': float(res_size.pvalues['rel_size']),
        'rel_location': float(res_loc.pvalues['rel_location'])
    },
    'std_coef': res_std.params.to_dict(),
    'std_pvalues': res_std.pvalues.to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
