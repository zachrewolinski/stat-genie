import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Construct predictors
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']  # positive => focal closer to its center

# Drop any missing
_df = _df.dropna(subset=['win', 'rel_size', 'size_ratio', 'loc_adv'])

# Standardized versions for effect size interpretation
_df['rel_size_z'] = (_df['rel_size'] - _df['rel_size'].mean()) / _df['rel_size'].std(ddof=0)
_df['loc_adv_z'] = (_df['loc_adv'] - _df['loc_adv'].mean()) / _df['loc_adv'].std(ddof=0)

# GLM with difference in size
X = sm.add_constant(_df[['rel_size', 'loc_adv']])
model = sm.GLM(_df['win'], X, family=sm.families.Binomial())
res = model.fit()

# GLM with standardized predictors
Xz = sm.add_constant(_df[['rel_size_z', 'loc_adv_z']])
model_z = sm.GLM(_df['win'], Xz, family=sm.families.Binomial())
res_z = model_z.fit()

# GLM with size ratio instead of difference
Xr = sm.add_constant(_df[['size_ratio', 'loc_adv']])
model_r = sm.GLM(_df['win'], Xr, family=sm.families.Binomial())
res_r = model_r.fit()

# Simple descriptive: win rate by location advantage sign
_df['focal_closer'] = _df['loc_adv'] > 0
win_rates = _df.groupby('focal_closer')['win'].mean().to_dict()
counts = _df.groupby('focal_closer')['win'].size().to_dict()

# Prepare summary dict
summary = {
    'n': int(len(_df)),
    'win_rate_overall': float(_df['win'].mean()),
    'glm_diff': {
        'params': res.params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
    },
    'glm_z': {
        'params': res_z.params.to_dict(),
        'pvalues': res_z.pvalues.to_dict(),
    },
    'glm_ratio': {
        'params': res_r.params.to_dict(),
        'pvalues': res_r.pvalues.to_dict(),
    },
    'win_rates_by_focal_closer': {
        'rates': win_rates,
        'counts': counts,
    }
}

print(json.dumps(summary, indent=2))
