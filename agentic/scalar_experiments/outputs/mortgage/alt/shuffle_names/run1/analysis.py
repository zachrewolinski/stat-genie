import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

print('shape', _df.shape)
print('columns', list(_df.columns))

# identify binary columns (0/1) and maybe 1/0 ints
binary_cols = []
for c in _df.columns:
    vals = _df[c].dropna().unique()
    if len(vals) <= 2 and set(vals).issubset({0,1,0.0,1.0}):
        binary_cols.append(c)
print('binary cols', binary_cols)

# compute means of binary columns
print('binary means')
for c in binary_cols:
    print(c, _df[c].mean())

# check accept/deny relationship
if 'accept' in _df.columns and 'deny' in _df.columns:
    # check complement
    comp = (_df['accept'] + _df['deny']).describe()
    print('accept+deny describe', comp)
    print('accept==1-deny', (_df['accept'] == 1-_df['deny']).mean())

# compute correlation with deny (if exists)
if 'deny' in _df.columns:
    corr = _df[binary_cols].corrwith(_df['deny'])
    print('corr with deny')
    print(corr.sort_values())

# compute simple chi-square for each binary vs deny
if 'deny' in _df.columns:
    print('chi-square for binary vs deny')
    for c in binary_cols:
        if c == 'deny':
            continue
        tbl = pd.crosstab(_df[c], _df['deny'])
        if tbl.shape == (2,2):
            chi2, p, dof, exp = stats.chi2_contingency(tbl)
            # effect size: difference in denial rates
            rate1 = tbl.loc[1,1] / tbl.loc[1].sum()
            rate0 = tbl.loc[0,1] / tbl.loc[0].sum()
            print(c, 'p=', p, 'deny_rate1', rate1, 'deny_rate0', rate0, 'diff', rate1-rate0)
        else:
            print(c, 'non-2x2 table', tbl.shape)

# logistic regression: deny ~ each binary
if 'deny' in _df.columns:
    print('logit deny ~ each binary')
    for c in binary_cols:
        if c == 'deny':
            continue
        X = sm.add_constant(_df[[c]])
        model = sm.Logit(_df['deny'], X, missing='drop')
        try:
            res = model.fit(disp=False)
            print(c, 'coef', res.params[c], 'p', res.pvalues[c])
        except Exception as e:
            print(c, 'fit error', e)

