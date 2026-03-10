import pandas as pd
import numpy as np
import statsmodels.api as sm


df=pd.read_csv('crofoot.csv')

# Define variables based on metadata
outcome = df['m_focal']  # 1 if focal won
size_focal = df['f_other']
size_other = df['win']
size_diff = size_focal - size_other
size_ratio = size_focal / size_other

# location: distance from each group's home range center
# smaller distance implies closer to own center; relative advantage for focal = other distance - focal distance
loc_adv = df['n_focal'] - df['m_other']

# Build dataframe
X = pd.DataFrame({
    'size_diff': size_diff,
    'loc_adv': loc_adv,
})
X = sm.add_constant(X)

model = sm.Logit(outcome, X)
try:
    res = model.fit(disp=False)
    print(res.summary())
except Exception as e:
    print('Logit failed:', e)
    res = None

# Also test size_ratio variant
X2 = pd.DataFrame({
    'size_ratio': size_ratio,
    'loc_adv': loc_adv,
})
X2 = sm.add_constant(X2)
model2 = sm.Logit(outcome, X2)
try:
    res2 = model2.fit(disp=False)
    print(res2.summary())
except Exception as e:
    print('Logit2 failed:', e)
    res2 = None

# Simple bivariate comparisons: mean size_diff by outcome, loc_adv by outcome
print('\nBivariate means:')
print(df.groupby('m_focal')[['f_other','win','m_other','n_focal']].mean())
print('mean size_diff by outcome', df.groupby('m_focal')[size_diff.name].mean())
print('mean loc_adv by outcome', df.groupby('m_focal')[loc_adv.name].mean())

# t-tests
from scipy import stats
for name, series in [('size_diff', size_diff), ('loc_adv', loc_adv)]:
    g0 = series[outcome==0]
    g1 = series[outcome==1]
    t,p = stats.ttest_ind(g1, g0, equal_var=False)
    print(name, 't', t, 'p', p)
