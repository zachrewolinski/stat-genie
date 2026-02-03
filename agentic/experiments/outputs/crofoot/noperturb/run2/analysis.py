import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Feature engineering
# Relative group size: focal size minus other size
# Contest location: positive if contest is closer to focal home range center than other
# Use continuous distance difference (other - focal), so positive means closer to focal

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparability
for col in ['rel_size', 'rel_dist']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression
X = df[['rel_size_z', 'rel_dist_z']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Print summary and key stats
print(result.summary())
print('\nOdds ratios (exp(coef)):')
print(np.exp(result.params))

# Simple interpretations: predicted probabilities for +/- 1 SD changes
base = pd.DataFrame({'const': [1], 'rel_size_z': [0], 'rel_dist_z': [0]})
plus_size = pd.DataFrame({'const': [1], 'rel_size_z': [1], 'rel_dist_z': [0]})
minus_size = pd.DataFrame({'const': [1], 'rel_size_z': [-1], 'rel_dist_z': [0]})
plus_dist = pd.DataFrame({'const': [1], 'rel_size_z': [0], 'rel_dist_z': [1]})
minus_dist = pd.DataFrame({'const': [1], 'rel_size_z': [0], 'rel_dist_z': [-1]})

pred_base = result.predict(base)[0]
pred_plus_size = result.predict(plus_size)[0]
pred_minus_size = result.predict(minus_size)[0]
pred_plus_dist = result.predict(plus_dist)[0]
pred_minus_dist = result.predict(minus_dist)[0]

print('\nPredicted win probability at mean predictors:', pred_base)
print('Predicted win probability +1 SD rel_size:', pred_plus_size)
print('Predicted win probability -1 SD rel_size:', pred_minus_size)
print('Predicted win probability +1 SD rel_dist:', pred_plus_dist)
print('Predicted win probability -1 SD rel_dist:', pred_minus_dist)

# Also test categorical location: focal closer indicator
# This is a sanity check

df['focal_closer'] = (df['dist_focal'] < df['dist_other']).astype(int)
X2 = sm.add_constant(df[['rel_size_z', 'focal_closer']])
model2 = sm.Logit(y, X2)
result2 = model2.fit(disp=False)

print('\nModel with focal_closer indicator:')
print(result2.summary())
print('\nOdds ratios (exp(coef)) for indicator model:')
print(np.exp(result2.params))
