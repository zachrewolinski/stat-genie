import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create relative predictors
# Relative group size: focal minus other

df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location: other distance minus focal distance (positive => focal closer to its center)

df['rel_location'] = df['dist_other'] - df['dist_focal']

# Prepare design matrix
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
y = df['win']

# Fit logistic regression (relative predictors)
model = sm.Logit(y, X)
res = model.fit(disp=False)

# Also fit with standardized predictors for effect comparison
X_std = df[['rel_size', 'rel_location']].copy()
X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
X_std = sm.add_constant(X_std)
model_std = sm.Logit(y, X_std)
res_std = model_std.fit(disp=False)

# Alternate model with raw sizes and distances
X_raw = df[['n_focal', 'n_other', 'dist_focal', 'dist_other']]
X_raw = sm.add_constant(X_raw)
model_raw = sm.Logit(y, X_raw)
res_raw = model_raw.fit(disp=False)

# Compute simple descriptive win rates by sign of predictors
for col in ['rel_size', 'rel_location']:
    df[col + '_pos'] = df[col] > 0

# Cross-tabs
win_rate_rel_size_pos = df.groupby('rel_size_pos')['win'].mean()
win_rate_rel_loc_pos = df.groupby('rel_location_pos')['win'].mean()

# Save results to a dict for convenience
results = {
    'n': len(df),
    'logit_params': res.params,
    'logit_pvalues': res.pvalues,
    'logit_conf_int': res.conf_int(),
    'logit_odds_ratios': np.exp(res.params),
    'logit_or_conf_int': np.exp(res.conf_int()),
    'logit_std_params': res_std.params,
    'logit_std_pvalues': res_std.pvalues,
    'logit_raw_params': res_raw.params,
    'logit_raw_pvalues': res_raw.pvalues,
    'win_rate_rel_size_pos': win_rate_rel_size_pos.to_dict(),
    'win_rate_rel_loc_pos': win_rate_rel_loc_pos.to_dict(),
}

# Print summary
print('N', results['n'])
print('\nLogit params')
print(results['logit_params'])
print('\nLogit p-values')
print(results['logit_pvalues'])
print('\nLogit odds ratios')
print(results['logit_odds_ratios'])
print('\nLogit OR 95% CI')
print(results['logit_or_conf_int'])
print('\nStandardized logit params')
print(results['logit_std_params'])
print('\nStandardized logit p-values')
print(results['logit_std_pvalues'])
print('\nRaw logit params (n_focal, n_other, dist_focal, dist_other)')
print(results['logit_raw_params'])
print('\nRaw logit p-values')
print(results['logit_raw_pvalues'])
print('\nWin rate when rel_size > 0 vs <= 0')
print(results['win_rate_rel_size_pos'])
print('\nWin rate when rel_location > 0 vs <= 0')
print(results['win_rate_rel_loc_pos'])
