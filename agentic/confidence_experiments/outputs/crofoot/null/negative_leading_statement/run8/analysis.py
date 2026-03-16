import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# Load data
_df = pd.read_csv('crofoot.csv')

# Create predictors
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_dist'] = _df['dist_other'] - _df['dist_focal']

# Standardize for interpretability in logistic regression
_df['rel_size_z'] = (_df['rel_size'] - _df['rel_size'].mean()) / _df['rel_size'].std(ddof=0)
_df['rel_dist_z'] = (_df['rel_dist'] - _df['rel_dist'].mean()) / _df['rel_dist'].std(ddof=0)

# Prepare design matrix
X = sm.add_constant(_df[['rel_size_z', 'rel_dist_z']])
y = _df['win']

result = None

try:
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
except PerfectSeparationError:
    result = None

summary = {}

if result is not None:
    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    summary = {
        'n': int(len(_df)),
        'coefficients': {
            'intercept': {
                'coef': float(params['const']),
                'p': float(pvalues['const']),
            },
            'rel_size_z': {
                'coef': float(params['rel_size_z']),
                'p': float(pvalues['rel_size_z']),
                'odds_ratio': float(odds_ratios['rel_size_z']),
                'or_ci_low': float(conf_odds.loc['rel_size_z', 0]),
                'or_ci_high': float(conf_odds.loc['rel_size_z', 1]),
            },
            'rel_dist_z': {
                'coef': float(params['rel_dist_z']),
                'p': float(pvalues['rel_dist_z']),
                'odds_ratio': float(odds_ratios['rel_dist_z']),
                'or_ci_low': float(conf_odds.loc['rel_dist_z', 0]),
                'or_ci_high': float(conf_odds.loc['rel_dist_z', 1]),
            },
        },
        'pseudo_r2': float(result.prsquared),
    }

# Also compute simple win-rate contrasts for context

# relative size grouped (focal larger, equal, smaller)
_df['size_cat'] = pd.cut(_df['rel_size'], bins=[-100, -1, 0, 100], labels=['focal_smaller', 'equal', 'focal_larger'])

size_rates = (
    _df.groupby('size_cat')['win']
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'win_rate'})
    .reset_index()
)

# relative distance grouped (closer to focal, equal, closer to other)
_df['dist_cat'] = pd.cut(_df['rel_dist'], bins=[-1e9, -1, 1, 1e9], labels=['closer_to_other', 'roughly_equal', 'closer_to_focal'])

# Note: a narrow bin for equality; with integer distances, exact equality is rare.

dist_rates = (
    _df.groupby('dist_cat')['win']
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'win_rate'})
    .reset_index()
)

# Save a compact JSON with key results
out = {
    'logit': summary,
    'size_rates': size_rates.to_dict(orient='records'),
    'dist_rates': dist_rates.to_dict(orient='records'),
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
