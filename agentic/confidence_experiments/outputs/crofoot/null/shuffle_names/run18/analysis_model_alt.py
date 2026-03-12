import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('crofoot.csv')

outcome = 'm_focal'

# alternate mapping (swap focal/other for size and distance)
size_focal = df['win']
size_other = df['f_other']
loc_focal = df['n_focal']
loc_other = df['m_other']

size_diff = size_focal - size_other
loc_adv = loc_other - loc_focal

X = pd.DataFrame({
    'size_diff_z': (size_diff - size_diff.mean()) / size_diff.std(ddof=0),
    'loc_adv_z': (loc_adv - loc_adv.mean()) / loc_adv.std(ddof=0),
})
X = sm.add_constant(X)

model = sm.Logit(df[outcome], X).fit(disp=False)
print(model.summary())

# univariate
for name, series in [('size_diff_z', X['size_diff_z']), ('loc_adv_z', X['loc_adv_z'])]:
    Xm = sm.add_constant(series)
    m = sm.Logit(df[outcome], Xm).fit(disp=False)
    print('\nUnivariate', name)
    print(m.summary())

print('\nWin rate by size_diff sign (focal bigger vs smaller)')
print(df.groupby(size_diff > 0)[outcome].mean())
print('\nWin rate by loc_adv sign (focal closer vs farther)')
print(df.groupby(loc_adv > 0)[outcome].mean())
