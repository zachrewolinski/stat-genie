import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Load data
csv_path = Path('crofoot.csv')

df = pd.read_csv(csv_path)

# Map columns based on info.json descriptions
# feature4: 1 if focal won contest, 0 if other won
# feature5: distance of focal from center of its home range
# feature6: distance of other from center of its home range
# feature7: number of individuals in focal group
# feature8: number of individuals in other group

# Derived predictors
# Relative group size (difference and ratio)
df['size_diff'] = df['feature7'] - df['feature8']
df['size_ratio'] = df['feature7'] / df['feature8']

# Relative contest location: positive means contest closer to focal (other is farther)
# If focal distance < other distance, then contest is closer to focal
# We use other_distance - focal_distance so positive favors focal

df['loc_diff'] = df['feature6'] - df['feature5']
# Also compute proportion closer to focal (0 to 1). lower means closer to focal
# Using focal distance / (focal + other)
df['loc_prop_focal'] = df['feature5'] / (df['feature5'] + df['feature6'])

# Response
y = df['feature4']

# Helper to fit logistic regression and output summary metrics

def fit_logit(X, name):
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X)
    res = model.fit(disp=False)
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues
    odds = np.exp(params)
    conf_odds = np.exp(conf)
    out = {
        'name': name,
        'n': int(res.nobs),
        'params': params.to_dict(),
        'pvalues': pvals.to_dict(),
        'odds_ratio': odds.to_dict(),
        'odds_ci': {k: [float(conf_odds.loc[k, 0]), float(conf_odds.loc[k, 1])] for k in conf_odds.index},
        'llf': float(res.llf),
        'aic': float(res.aic),
    }
    return out

# Models
models = []

# Model with size_diff and loc_diff
models.append(fit_logit(df[['size_diff', 'loc_diff']], 'size_diff + loc_diff'))

# Model with size_ratio and loc_diff
models.append(fit_logit(df[['size_ratio', 'loc_diff']], 'size_ratio + loc_diff'))

# Model with size_diff and loc_prop_focal (lower = closer to focal)
models.append(fit_logit(df[['size_diff', 'loc_prop_focal']], 'size_diff + loc_prop_focal'))

# Model with size_ratio and loc_prop_focal
models.append(fit_logit(df[['size_ratio', 'loc_prop_focal']], 'size_ratio + loc_prop_focal'))

# Single predictor models
models.append(fit_logit(df[['size_diff']], 'size_diff only'))
models.append(fit_logit(df[['loc_diff']], 'loc_diff only'))

# Save results
results = {
    'descriptives': {
        'n': int(df.shape[0]),
        'win_rate': float(df['feature4'].mean()),
        'size_diff_mean': float(df['size_diff'].mean()),
        'size_diff_std': float(df['size_diff'].std()),
        'loc_diff_mean': float(df['loc_diff'].mean()),
        'loc_diff_std': float(df['loc_diff'].std()),
    },
    'models': models,
}

import json
Path('analysis_results.json').write_text(json.dumps(results, indent=2))

# Print concise summary for quick inspection
print('N:', results['descriptives']['n'])
print('Win rate:', results['descriptives']['win_rate'])
for m in models:
    print('\nModel:', m['name'])
    for k, v in m['params'].items():
        print(f"  {k}: coef={v:.4f}, p={m['pvalues'][k]:.4g}, OR={m['odds_ratio'][k]:.3f}")
