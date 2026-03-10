import pandas as pd
import numpy as np
import statsmodels.api as sm

pd.set_option('display.width', 160)

# Load data
_df = pd.read_csv('crofoot.csv')

# Map variables based on consistency checks
# outcome: focal win (0/1)
y = _df['m_focal']

# group sizes
focal_size = _df['f_other']  # dist_focal + other
other_size = _df['win']      # focal + f_focal

# contest location: distance from home range centers
# m_other and n_focal are the only large-valued distance columns
focal_dist = _df['m_other']
other_dist = _df['n_focal']

# predictors
_df = _df.copy()
_df['rel_size'] = focal_size - other_size
_df['rel_loc'] = focal_dist - other_dist

# Standardize predictors for comparability
_df['rel_size_z'] = (_df['rel_size'] - _df['rel_size'].mean()) / _df['rel_size'].std(ddof=0)
_df['rel_loc_z'] = (_df['rel_loc'] - _df['rel_loc'].mean()) / _df['rel_loc'].std(ddof=0)

X = sm.add_constant(_df[['rel_size_z', 'rel_loc_z']])
model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())

# Also fit separate models for each predictor
for col in ['rel_size_z', 'rel_loc_z']:
    X1 = sm.add_constant(_df[[col]])
    res1 = sm.Logit(y, X1).fit(disp=False)
    print('\nSingle predictor:', col)
    print(res1.summary())

# Compute odds ratios for 1 SD increase
params = result.params
conf = result.conf_int()

odds = np.exp(params)
conf_odds = np.exp(conf)

print('\nOdds ratios (1 SD):')
print(pd.DataFrame({'odds_ratio': odds, 'ci_low': conf_odds[0], 'ci_high': conf_odds[1]}))

# Basic descriptive stats
print('\nRel size mean/std:', _df['rel_size'].mean(), _df['rel_size'].std(ddof=0))
print('Rel loc mean/std:', _df['rel_loc'].mean(), _df['rel_loc'].std(ddof=0))
print('Win rate:', y.mean())

