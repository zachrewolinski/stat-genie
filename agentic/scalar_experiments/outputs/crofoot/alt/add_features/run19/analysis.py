import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Focus on relevant columns
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Compute relative measures
# Relative group size: focal - other
# Relative location: other distance - focal distance (positive => focal closer to its center)

df = df[cols].dropna().copy()
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_other'] - df['dist_focal']

# Logistic regression with relative measures
X = df[['rel_size', 'rel_dist']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also fit model with absolute distances and sizes for sensitivity
X2 = df[['n_focal', 'n_other', 'dist_focal', 'dist_other']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2)
result2 = model2.fit(disp=False)

# Summaries
print('n=', len(df))
print('\nRelative model coefficients:')
print(result.params)
print('\nRelative model p-values:')
print(result.pvalues)
print('\nRelative model odds ratios:')
print(np.exp(result.params))

print('\nAbsolute model coefficients:')
print(result2.params)
print('\nAbsolute model p-values:')
print(result2.pvalues)

# Effect sizes at 1 SD change for rel_dist and rel_size (odds ratio)
rel_size_sd = df['rel_size'].std()
rel_dist_sd = df['rel_dist'].std()
print('\nSD rel_size:', rel_size_sd, 'SD rel_dist:', rel_dist_sd)
print('OR per 1 SD rel_size:', np.exp(result.params['rel_size'] * rel_size_sd))
print('OR per 1 SD rel_dist:', np.exp(result.params['rel_dist'] * rel_dist_sd))

# Simple group comparisons
# Compare win rate when focal larger vs smaller
larger = df[df['rel_size'] > 0]['win']
smaller = df[df['rel_size'] < 0]['win']
print('\nWin rate when focal larger:', larger.mean())
print('Win rate when focal smaller:', smaller.mean())

# Location: focal closer (rel_dist > 0)
closer = df[df['rel_dist'] > 0]['win']
farther = df[df['rel_dist'] < 0]['win']
print('\nWin rate when focal closer to its center:', closer.mean())
print('Win rate when focal farther:', farther.mean())
