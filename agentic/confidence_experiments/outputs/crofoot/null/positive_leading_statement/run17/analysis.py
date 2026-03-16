import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Create predictors
# Relative group size (difference)
df['size_diff'] = df['n_focal'] - df['n_other']
# Relative contest location: positive when contest is closer to focal home-range center
# (other is farther from its center than focal)
df['dist_diff'] = df['dist_other'] - df['dist_focal']

# Response
y = df['win']

# Add constant
X = df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X).fit(disp=False)

# Also fit bivariate models for robustness
model_size = sm.Logit(y, sm.add_constant(df[['size_diff']])).fit(disp=False)
model_dist = sm.Logit(y, sm.add_constant(df[['dist_diff']])).fit(disp=False)

# Compute odds ratios and 95% CI
def odds_ratio_ci(mod):
    params = mod.params
    conf = mod.conf_int()
    or_vals = np.exp(params)
    or_ci = np.exp(conf)
    return or_vals, or_ci

or_full, or_ci_full = odds_ratio_ci(model)
or_size, or_ci_size = odds_ratio_ci(model_size)
or_dist, or_ci_dist = odds_ratio_ci(model_dist)

# Save key results
results = {
    'n': int(df.shape[0]),
    'full_model': {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'odds_ratio': or_full.to_dict(),
        'odds_ratio_ci': {k: [float(or_ci_full.loc[k, 0]), float(or_ci_full.loc[k, 1])] for k in or_ci_full.index},
        'pseudo_r2': float(model.prsquared),
    },
    'size_only': {
        'params': model_size.params.to_dict(),
        'pvalues': model_size.pvalues.to_dict(),
        'odds_ratio': or_size.to_dict(),
        'odds_ratio_ci': {k: [float(or_ci_size.loc[k, 0]), float(or_ci_size.loc[k, 1])] for k in or_ci_size.index},
        'pseudo_r2': float(model_size.prsquared),
    },
    'dist_only': {
        'params': model_dist.params.to_dict(),
        'pvalues': model_dist.pvalues.to_dict(),
        'odds_ratio': or_dist.to_dict(),
        'odds_ratio_ci': {k: [float(or_ci_dist.loc[k, 0]), float(or_ci_dist.loc[k, 1])] for k in or_ci_dist.index},
        'pseudo_r2': float(model_dist.prsquared),
    },
    'descriptive': {
        'win_rate': float(df['win'].mean()),
        'size_diff_mean': float(df['size_diff'].mean()),
        'dist_diff_mean': float(df['dist_diff'].mean()),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(model.summary())
print('\nBivariate size-only model:')
print(model_size.summary())
print('\nBivariate dist-only model:')
print(model_dist.summary())
