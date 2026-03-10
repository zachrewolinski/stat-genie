import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('crofoot.csv')

# Derived variables
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['location_diff'] = _df['dist_other'] - _df['dist_focal']  # positive means contest closer to focal home range

# Standardize continuous predictors for interpretability
for col in ['size_diff', 'size_ratio', 'location_diff']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Logistic regression using size_diff and location_diff
X = _df[['size_diff_z', 'location_diff_z']]
X = sm.add_constant(X)
y = _df['win']

model = sm.Logit(y, X).fit(disp=False)

# Alternative model using size_ratio
X2 = _df[['size_ratio_z', 'location_diff_z']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2).fit(disp=False)

# Likelihood ratio test against intercept-only model
X0 = sm.add_constant(pd.DataFrame({'const': np.ones(len(_df))}))
model0 = sm.Logit(y, X0).fit(disp=False)

llf_full = model.llf
llf_null = model0.llf
lr_stat = 2 * (llf_full - llf_null)
lr_p = stats.chi2.sf(lr_stat, df=2)

# Compute basic effect sizes: odds ratios for 1 SD change
params = model.params
conf = model.conf_int()
ors = np.exp(params)
conf_or = np.exp(conf)

# Simple descriptive comparisons
_df['size_advantage'] = np.where(_df['n_focal'] > _df['n_other'], 'focal_larger',
                                 np.where(_df['n_focal'] < _df['n_other'], 'focal_smaller', 'equal'))
_df['location_advantage'] = np.where(_df['dist_focal'] < _df['dist_other'], 'focal_closer',
                                     np.where(_df['dist_focal'] > _df['dist_other'], 'other_closer', 'equal'))

size_win_rates = _df.groupby('size_advantage')['win'].mean().to_dict()
loc_win_rates = _df.groupby('location_advantage')['win'].mean().to_dict()

# Fisher's exact test for binary advantage (exclude ties)
size_bin = _df[_df['size_advantage'] != 'equal'].copy()
size_bin['focal_larger'] = (size_bin['size_advantage'] == 'focal_larger').astype(int)
size_table = pd.crosstab(size_bin['focal_larger'], size_bin['win'])
size_fisher = stats.fisher_exact(size_table) if size_table.shape == (2, 2) else (np.nan, np.nan)

loc_bin = _df[_df['location_advantage'] != 'equal'].copy()
loc_bin['focal_closer'] = (loc_bin['location_advantage'] == 'focal_closer').astype(int)
loc_table = pd.crosstab(loc_bin['focal_closer'], loc_bin['win'])
loc_fisher = stats.fisher_exact(loc_table) if loc_table.shape == (2, 2) else (np.nan, np.nan)

# Save results
results = {
    'n': len(_df),
    'coef': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'odds_ratios': ors.to_dict(),
    'or_ci_low': conf_or[0].to_dict(),
    'or_ci_high': conf_or[1].to_dict(),
    'lr_stat': float(lr_stat),
    'lr_p': float(lr_p),
    'model2_coef': model2.params.to_dict(),
    'model2_pvalues': model2.pvalues.to_dict(),
    'size_win_rates': size_win_rates,
    'location_win_rates': loc_win_rates,
    'size_fisher_oddsratio': float(size_fisher[0]) if size_fisher[0] is not np.nan else None,
    'size_fisher_p': float(size_fisher[1]) if size_fisher[1] is not np.nan else None,
    'location_fisher_oddsratio': float(loc_fisher[0]) if loc_fisher[0] is not np.nan else None,
    'location_fisher_p': float(loc_fisher[1]) if loc_fisher[1] is not np.nan else None,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
