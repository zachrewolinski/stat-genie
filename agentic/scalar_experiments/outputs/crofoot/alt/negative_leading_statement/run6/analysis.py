import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('crofoot.csv')

# Create predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['dist_diff'] = _df['dist_focal'] - _df['dist_other']
_df['dist_ratio'] = _df['dist_focal'] / _df['dist_other']

# Basic summaries
summary = {
    'n': len(_df),
    'win_rate': _df['win'].mean(),
    'size_diff_mean': _df['size_diff'].mean(),
    'dist_diff_mean': _df['dist_diff'].mean(),
}

# Logistic regression with size_diff and dist_diff
model1 = smf.logit('win ~ size_diff + dist_diff', data=_df).fit(disp=False)

# Logistic regression with size_ratio and dist_ratio
model2 = smf.logit('win ~ size_ratio + dist_ratio', data=_df).fit(disp=False)

# Logistic regression with dist_focal and dist_other separately
model3 = smf.logit('win ~ size_diff + dist_focal + dist_other', data=_df).fit(disp=False)

# Simple tests: compare size_diff and dist_diff between wins and losses
win_group = _df[_df['win'] == 1]
lose_group = _df[_df['win'] == 0]

t_size = stats.ttest_ind(win_group['size_diff'], lose_group['size_diff'], equal_var=False)

t_dist = stats.ttest_ind(win_group['dist_diff'], lose_group['dist_diff'], equal_var=False)

# Compute effect sizes (Cohen's d)

def cohens_d(x, y):
    nx = len(x); ny = len(y)
    vx = x.var(ddof=1); vy = y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)


size_d = cohens_d(win_group['size_diff'], lose_group['size_diff'])
dist_d = cohens_d(win_group['dist_diff'], lose_group['dist_diff'])

# Save results to a dict
results = {
    'summary': summary,
    'model1_params': model1.params.to_dict(),
    'model1_pvalues': model1.pvalues.to_dict(),
    'model1_aic': float(model1.aic),
    'model2_params': model2.params.to_dict(),
    'model2_pvalues': model2.pvalues.to_dict(),
    'model2_aic': float(model2.aic),
    'model3_params': model3.params.to_dict(),
    'model3_pvalues': model3.pvalues.to_dict(),
    'model3_aic': float(model3.aic),
    't_size': {'stat': float(t_size.statistic), 'p': float(t_size.pvalue)},
    't_dist': {'stat': float(t_dist.statistic), 'p': float(t_dist.pvalue)},
    'cohens_d': {'size_diff': float(size_d), 'dist_diff': float(dist_d)},
}

# Print for inspection
import json
print(json.dumps(results, indent=2))
