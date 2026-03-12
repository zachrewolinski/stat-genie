import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('crofoot.csv')

# Create relative measures
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_males'] = df['m_focal'] - df['m_other']
df['rel_females'] = df['f_focal'] - df['f_other']

# Location advantage: positive means focal closer to its own center than other is to theirs
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Logistic regression: win ~ rel_size + loc_adv
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X).fit(disp=False)

# Also check model with standardized predictors to compare effect sizes
Xz = df[['rel_size', 'loc_adv']].apply(lambda c: (c - c.mean()) / c.std())
Xz = sm.add_constant(Xz)
model_z = sm.Logit(df['win'], Xz).fit(disp=False)

# Single-predictor models
X_size = sm.add_constant(df[['rel_size']])
model_size = sm.Logit(df['win'], X_size).fit(disp=False)

X_loc = sm.add_constant(df[['loc_adv']])
model_loc = sm.Logit(df['win'], X_loc).fit(disp=False)

# Simple group comparisons for win by location advantage and size advantage
df['size_adv'] = (df['rel_size'] > 0).astype(int)
df['loc_adv_bin'] = (df['loc_adv'] > 0).astype(int)

# Win rates
win_rate_size_adv = df.groupby('size_adv')['win'].mean()
win_rate_loc_adv = df.groupby('loc_adv_bin')['win'].mean()

# Chi-square tests for association
ct_size = pd.crosstab(df['size_adv'], df['win'])
ct_loc = pd.crosstab(df['loc_adv_bin'], df['win'])
chi_size = stats.chi2_contingency(ct_size)
chi_loc = stats.chi2_contingency(ct_loc)

print('N', len(df))
print('Logit win ~ rel_size + loc_adv')
print(model.summary())
print('\nLogit (standardized predictors)')
print(model_z.summary())
print('\nLogit win ~ rel_size')
print(model_size.summary())
print('\nLogit win ~ loc_adv')
print(model_loc.summary())
print('\nWin rates by size_adv (rel_size>0)')
print(win_rate_size_adv)
print('Chi-square size_adv vs win:', chi_size)
print('\nWin rates by loc_adv (loc_adv>0)')
print(win_rate_loc_adv)
print('Chi-square loc_adv vs win:', chi_loc)
