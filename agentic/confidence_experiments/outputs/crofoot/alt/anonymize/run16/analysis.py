import json
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data

df = pd.read_csv('crofoot.csv')

# Rename for clarity

df = df.rename(columns={
    'feature4': 'focal_win',
    'feature5': 'focal_dist',
    'feature6': 'other_dist',
    'feature7': 'focal_size',
    'feature8': 'other_size',
})

# Derived predictors

df['rel_group_size'] = df['focal_size'] - df['other_size']
# Positive when contest is closer to focal group (other further from its center)

df['location_advantage'] = df['other_dist'] - df['focal_dist']

# Logistic regression with both predictors

X = df[['rel_group_size', 'location_advantage']]
X = sm.add_constant(X)
model = sm.Logit(df['focal_win'], X)
res = model.fit(disp=False)

# Single-predictor models for robustness

X_size = sm.add_constant(df[['rel_group_size']])
res_size = sm.Logit(df['focal_win'], X_size).fit(disp=False)

X_loc = sm.add_constant(df[['location_advantage']])
res_loc = sm.Logit(df['focal_win'], X_loc).fit(disp=False)

# Collect stats

out = {
    'n': int(len(df)),
    'wins': int(df['focal_win'].sum()),
    'win_rate': float(df['focal_win'].mean()),
    'coef': res.params.to_dict(),
    'pvalues': res.pvalues.to_dict(),
    'odds_ratios': {k: float(np.exp(v)) for k, v in res.params.to_dict().items()},
    'coef_size_only': res_size.params.to_dict(),
    'pvalues_size_only': res_size.pvalues.to_dict(),
    'coef_loc_only': res_loc.params.to_dict(),
    'pvalues_loc_only': res_loc.pvalues.to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
