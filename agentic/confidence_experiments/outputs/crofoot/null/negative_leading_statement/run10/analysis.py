import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Derive predictors
_df['rel_size'] = _df['n_focal'] - _df['n_other']
# Positive rel_loc means focal is closer to its own home-range center than the other group is to its own
_df['rel_loc'] = _df['dist_other'] - _df['dist_focal']

# Standardize for interpretability
for col in ['rel_size', 'rel_loc']:
    _df[f'{col}_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Fit logistic regression with cluster-robust SE by dyad (accounts for repeated dyads)
X = sm.add_constant(_df[['rel_size_z', 'rel_loc_z']])
robust = sm.Logit(_df['win'], X).fit(
    disp=0,
    cov_type='cluster',
    cov_kwds={'groups': _df['dyad']},
)

# Single-predictor models for sensitivity
X_size = sm.add_constant(_df[['rel_size_z']])
robust_size = sm.Logit(_df['win'], X_size).fit(
    disp=0,
    cov_type='cluster',
    cov_kwds={'groups': _df['dyad']},
)

X_loc = sm.add_constant(_df[['rel_loc_z']])
robust_loc = sm.Logit(_df['win'], X_loc).fit(
    disp=0,
    cov_type='cluster',
    cov_kwds={'groups': _df['dyad']},
)

# Compute odds ratios for 1 SD change
params = robust.params
ses = robust.bse
pvals = robust.pvalues

odds_ratios = np.exp(params)

# Build a small report dict for printing
report = {
    'n': int(_df.shape[0]),
    'rel_size_mean': float(_df['rel_size'].mean()),
    'rel_loc_mean': float(_df['rel_loc'].mean()),
    'logit_main': {
        'coef': params.to_dict(),
        'se': ses.to_dict(),
        'pval': pvals.to_dict(),
        'odds_ratio': odds_ratios.to_dict(),
    },
    'logit_size_only': {
        'coef': robust_size.params.to_dict(),
        'se': robust_size.bse.to_dict(),
        'pval': robust_size.pvalues.to_dict(),
    },
    'logit_loc_only': {
        'coef': robust_loc.params.to_dict(),
        'se': robust_loc.bse.to_dict(),
        'pval': robust_loc.pvalues.to_dict(),
    },
}

print(json.dumps(report, indent=2, sort_keys=True))
