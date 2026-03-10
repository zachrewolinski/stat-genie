import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Map variables based on value ranges/metadata consistency
# outcome: m_focal (binary win for focal)
# group sizes: f_other (focal size) and win (other size)
# distances: m_other (focal distance from center), n_focal (other distance from center)

outcome = 'm_focal'
size_focal = df['f_other']
size_other = df['win']
loc_focal = df['m_other']
loc_other = df['n_focal']

# Relative measures
size_diff = size_focal - size_other
loc_adv = loc_other - loc_focal  # positive if other is farther from its center than focal

# Standardize predictors
X = pd.DataFrame({
    'size_diff_z': (size_diff - size_diff.mean()) / size_diff.std(ddof=0),
    'loc_adv_z': (loc_adv - loc_adv.mean()) / loc_adv.std(ddof=0),
})
X = sm.add_constant(X)

model = sm.Logit(df[outcome], X).fit(disp=False)
print(model.summary())

# Also fit univariate models for context
for name, series in [('size_diff_z', X['size_diff_z']), ('loc_adv_z', X['loc_adv_z'])]:
    Xm = sm.add_constant(series)
    m = sm.Logit(df[outcome], Xm).fit(disp=False)
    print('\nUnivariate', name)
    print(m.summary())

# Simple descriptive: win rate by size_diff and loc_adv sign
print('\nWin rate by size_diff sign (focal bigger vs smaller)')
print(df.groupby(size_diff > 0)[outcome].mean())
print('\nWin rate by loc_adv sign (focal closer vs farther)')
print(df.groupby(loc_adv > 0)[outcome].mean())
