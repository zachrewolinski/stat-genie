import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Variables based on metadata
# Outcome: m_focal (1 focal wins)
# Relative group size: f_other (focal group size) vs win (other group size)
# Location: m_other (distance focal from its home range center), n_focal (distance other from its center)

_df = _df.copy()
_df['size_diff'] = _df['f_other'] - _df['win']
_df['size_ratio'] = _df['f_other'] / _df['win']
_df['location_adv'] = _df['n_focal'] - _df['m_other']  # positive => contest farther from other group's center

print('Outcome mean (focal win rate):', _df['m_focal'].mean())

# Logistic regression with size_diff and location_adv
X = _df[['size_diff', 'location_adv']]
X = sm.add_constant(X)
model = sm.Logit(_df['m_focal'], X)
res = model.fit(disp=False)
print('\nLogit: m_focal ~ size_diff + location_adv')
print(res.summary())

# Alternative: size_ratio instead of size_diff
X2 = _df[['size_ratio', 'location_adv']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(_df['m_focal'], X2)
res2 = model2.fit(disp=False)
print('\nLogit: m_focal ~ size_ratio + location_adv')
print(res2.summary())

# Simple correlations (point-biserial equivalent) for context
print('\nCorrelation with outcome:')
for col in ['size_diff', 'size_ratio', 'location_adv', 'm_other', 'n_focal']:
    corr = np.corrcoef(_df['m_focal'], _df[col])[0,1]
    print(f'{col}: {corr:.3f}')

# Grouped win rates by size_diff
print('\nWin rate by size_diff:')
print(_df.groupby('size_diff')['m_focal'].mean())

# Grouped win rates by location_adv quartiles
_df['loc_q'] = pd.qcut(_df['location_adv'], 4, duplicates='drop')
print('\nWin rate by location_adv quartile:')
print(_df.groupby('loc_q')['m_focal'].mean())
