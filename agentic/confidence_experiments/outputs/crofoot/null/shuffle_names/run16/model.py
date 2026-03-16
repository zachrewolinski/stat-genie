import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Map variables based on metadata ranges
# Outcome: focal won (binary)
_df['win_focal'] = _df['m_focal']

# Group sizes (5-13): focal and other
_df['size_focal'] = _df['f_other']
_df['size_other'] = _df['win']

# Distances to home range center (meters)
_df['dist_focal_center'] = _df['m_other']
_df['dist_other_center'] = _df['n_focal']

# Relative predictors
_df['rel_size'] = _df['size_focal'] - _df['size_other']
_df['rel_dist'] = _df['dist_other_center'] - _df['dist_focal_center']
# rel_dist > 0 => contest closer to focal center than other (other is farther)

# Logistic regression with relative size and location
X = _df[['rel_size','rel_dist']]
X = sm.add_constant(X)
y = _df['win_focal']
model = sm.Logit(y, X).fit(disp=False)

print('Logit with rel_size and rel_dist')
print(model.summary())

# Also test each predictor separately for context
X_size = sm.add_constant(_df[['rel_size']])
model_size = sm.Logit(y, X_size).fit(disp=False)
print('\nLogit with rel_size only')
print(model_size.summary())

X_dist = sm.add_constant(_df[['rel_dist']])
model_dist = sm.Logit(y, X_dist).fit(disp=False)
print('\nLogit with rel_dist only')
print(model_dist.summary())

# Simple descriptive probabilities by sign of rel_size and rel_dist
for col in ['rel_size','rel_dist']:
    pos = _df[_df[col] > 0]
    neg = _df[_df[col] < 0]
    zero = _df[_df[col] == 0]
    print(f"\n{col} groups: >0 n={len(pos)} win_rate={pos['win_focal'].mean():.3f}; <0 n={len(neg)} win_rate={neg['win_focal'].mean():.3f}; =0 n={len(zero)} win_rate={zero['win_focal'].mean():.3f}")

# Correlation for intuition
print('\nCorrelations:')
print(_df[['win_focal','rel_size','rel_dist']].corr())
