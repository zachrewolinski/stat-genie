import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
_df = pd.read_csv('crofoot.csv')

# Create relative size and location metrics
# relative size difference (focal - other)
_df['rel_size'] = _df['n_focal'] - _df['n_other']
# location advantage: positive if contest closer to focal home range center
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']

# Basic summaries
summary = {
    'n_rows': len(_df),
    'win_rate': _df['win'].mean(),
    'rel_size_mean': _df['rel_size'].mean(),
    'loc_adv_mean': _df['loc_adv'].mean(),
}

# Logistic regression with both predictors
X = _df[['rel_size', 'loc_adv']].copy()
X = sm.add_constant(X)
model = sm.Logit(_df['win'], X)
res = model.fit(disp=False)

# Also standardized predictors to compare effect sizes
Xz = _df[['rel_size', 'loc_adv']].copy()
Xz = (Xz - Xz.mean()) / Xz.std(ddof=0)
Xz = sm.add_constant(Xz)
res_z = sm.Logit(_df['win'], Xz).fit(disp=False)

# Simple binned comparisons for interpretability
_df['rel_size_cat'] = pd.cut(_df['rel_size'], bins=[-10, -1, 1, 10], labels=['smaller', 'similar', 'larger'])
_df['loc_adv_cat'] = pd.cut(_df['loc_adv'], bins=[-1000, -1, 1, 1000], labels=['closer_to_other','about_equal','closer_to_focal'])

win_by_size = _df.groupby('rel_size_cat')['win'].mean().to_dict()
win_by_loc = _df.groupby('loc_adv_cat')['win'].mean().to_dict()

# Output key stats
out = {
    'summary': summary,
    'logit_params': res.params.to_dict(),
    'logit_pvalues': res.pvalues.to_dict(),
    'logit_conf_int': {k: [float(v) for v in res.conf_int().loc[k].tolist()] for k in res.params.index},
    'logit_n': int(res.nobs),
    'logit_llf': float(res.llf),
    'logit_prsquared': float(res.prsquared),
    'logit_z_params': res_z.params.to_dict(),
    'logit_z_pvalues': res_z.pvalues.to_dict(),
    'win_by_size': {k: (float(v) if pd.notna(v) else None) for k, v in win_by_size.items()},
    'win_by_loc': {k: (float(v) if pd.notna(v) else None) for k, v in win_by_loc.items()},
}

with open('analysis_output.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
