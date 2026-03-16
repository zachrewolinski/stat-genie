import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Compute predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']
# location advantage: positive if contest is closer to focal home range center
_df['loc_diff'] = _df['dist_other'] - _df['dist_focal']

# Standardize for interpretability
_df['size_diff_z'] = (_df['size_diff'] - _df['size_diff'].mean()) / _df['size_diff'].std(ddof=0)
_df['loc_diff_z'] = (_df['loc_diff'] - _df['loc_diff'].mean()) / _df['loc_diff'].std(ddof=0)

# Prepare model
X = _df[['size_diff_z','loc_diff_z']]
X = sm.add_constant(X)

y = _df['win']

model = sm.Logit(y, X)
res = model.fit(disp=False)

# Also run model with unstandardized for effect sign
X2 = sm.add_constant(_df[['size_diff','loc_diff']])
res2 = sm.Logit(y, X2).fit(disp=False)

# basic stats
summary = {
    'n': len(_df),
    'wins': int(_df['win'].sum()),
    'losses': int((1-_df['win']).sum()),
    'size_diff_mean': float(_df['size_diff'].mean()),
    'loc_diff_mean': float(_df['loc_diff'].mean())
}

# predicted probabilities for high vs low
# compute predicted p at mean with +1 SD and -1 SD for each predictor
mean_vec = X[['size_diff_z','loc_diff_z']].mean()

def pred_at(size_z, loc_z):
    v = pd.DataFrame({'const':[1.0],'size_diff_z':[size_z],'loc_diff_z':[loc_z]})
    return float(res.predict(v)[0])

preds = {
    'p_mean': pred_at(mean_vec['size_diff_z'], mean_vec['loc_diff_z']),
    'p_size_plus1': pred_at(1.0, mean_vec['loc_diff_z']),
    'p_size_minus1': pred_at(-1.0, mean_vec['loc_diff_z']),
    'p_loc_plus1': pred_at(mean_vec['size_diff_z'], 1.0),
    'p_loc_minus1': pred_at(mean_vec['size_diff_z'], -1.0),
}

# Save key outputs
out = {
    'summary': summary,
    'coef_z': res.params.to_dict(),
    'pvalues_z': res.pvalues.to_dict(),
    'coef_raw': res2.params.to_dict(),
    'pvalues_raw': res2.pvalues.to_dict(),
    'preds': preds,
    'llf': float(res.llf),
    'llnull': float(res.llnull),
    'prsquared': float(res.prsquared)
}

import json
with open('analysis_results.json','w') as f:
    json.dump(out,f,indent=2)

print(json.dumps(out,indent=2))
