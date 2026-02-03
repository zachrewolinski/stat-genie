import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Outcome: focal wins (1) vs other wins (0)
# Based on metadata: m_focal is win indicator
if 'm_focal' not in df.columns:
    raise ValueError('Expected m_focal column for win indicator')

# Relative group size: focal group size minus other group size
# Based on metadata: f_other = focal group size, win = other group size
if not {'f_other', 'win'}.issubset(df.columns):
    raise ValueError('Expected f_other and win columns for group sizes')

df['rel_group_size'] = df['f_other'] - df['win']

# Contest location: relative distance from home-range centers
# Based on metadata: m_other = focal distance, n_focal = other distance
if not {'m_other', 'n_focal'}.issubset(df.columns):
    raise ValueError('Expected m_other and n_focal columns for distances')

df['rel_distance'] = df['m_other'] - df['n_focal']

# Basic checks
print('Rows:', len(df))
print('Win rate (focal):', df['m_focal'].mean())
print('Rel group size (mean, std):', df['rel_group_size'].mean(), df['rel_group_size'].std())
print('Rel distance (mean, std):', df['rel_distance'].mean(), df['rel_distance'].std())

# Logistic regression
X = df[['rel_group_size', 'rel_distance']]
X = sm.add_constant(X)
y = df['m_focal']

model = sm.Logit(y, X).fit(disp=False)
print('\nLogit coefficients:')
print(model.params)
print('\nLogit summary:')
print(model.summary())

# Odds ratios
odds_ratios = np.exp(model.params)
print('\nOdds ratios:')
print(odds_ratios)

# Simple stratified summaries
# Win rate by relative size categories
bins = [-np.inf, -1, 0, 1, np.inf]
labels = ['<=-2', '-1 to 0', '1', '>=2']
# Make bins more meaningful: difference values are integers
# Use custom bins that separate negative, zero, positive
bins = [-np.inf, -1, 0, np.inf]
labels = ['focal smaller', 'equal size', 'focal larger']

df['size_cat'] = pd.cut(df['rel_group_size'], bins=bins, labels=labels)
print('\nWin rate by relative size category:')
print(df.groupby('size_cat')['m_focal'].mean())

# Win rate by relative distance (focal closer vs other closer)
# focal closer if rel_distance < 0

df['location_cat'] = np.where(df['rel_distance'] < 0, 'focal closer', 'other closer')
print('\nWin rate by relative location:')
print(df.groupby('location_cat')['m_focal'].mean())
