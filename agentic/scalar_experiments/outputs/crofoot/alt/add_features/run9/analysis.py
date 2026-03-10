import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'crofoot.csv'

df = pd.read_csv(csv_path)

# Keep relevant columns
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']

# Drop rows with missing in these columns
sub = df[cols].dropna().copy()

# Derived variables
sub['rel_size'] = sub['n_focal'] - sub['n_other']
sub['rel_dist'] = sub['dist_other'] - sub['dist_focal']  # positive: contest closer to focal center than to other

# Standardize predictors for interpretability
for col in ['rel_size', 'rel_dist']:
    sub[f'z_{col}'] = (sub[col] - sub[col].mean()) / sub[col].std(ddof=0)

# Logistic regression with raw predictors
model_raw = smf.logit('win ~ rel_size + rel_dist', data=sub).fit(disp=False)

# Logistic regression with standardized predictors
model_std = smf.logit('win ~ z_rel_size + z_rel_dist', data=sub).fit(disp=False)

# Alternative model with dist_focal and dist_other
model_dist = smf.logit('win ~ dist_focal + dist_other', data=sub).fit(disp=False)

# Simple bivariate correlations (point-biserial as Pearson on binary)
corr_rel_size = np.corrcoef(sub['win'], sub['rel_size'])[0,1]
corr_rel_dist = np.corrcoef(sub['win'], sub['rel_dist'])[0,1]

results = {
    'n_rows': int(sub.shape[0]),
    'rel_size_mean': float(sub['rel_size'].mean()),
    'rel_dist_mean': float(sub['rel_dist'].mean()),
    'model_raw': {
        'params': model_raw.params.to_dict(),
        'pvalues': model_raw.pvalues.to_dict(),
        'llf': float(model_raw.llf),
        'pseudo_r2': float(model_raw.prsquared),
    },
    'model_std': {
        'params': model_std.params.to_dict(),
        'pvalues': model_std.pvalues.to_dict(),
        'llf': float(model_std.llf),
        'pseudo_r2': float(model_std.prsquared),
    },
    'model_dist': {
        'params': model_dist.params.to_dict(),
        'pvalues': model_dist.pvalues.to_dict(),
        'llf': float(model_dist.llf),
        'pseudo_r2': float(model_dist.prsquared),
    },
    'corr_rel_size': float(corr_rel_size),
    'corr_rel_dist': float(corr_rel_dist),
}

print(json.dumps(results, indent=2))
