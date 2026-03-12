import pandas as pd
import numpy as np

_df=pd.read_csv('mortgage.csv')

binary_cols=[c for c in _df.columns if set(_df[c].dropna().unique()).issubset({0,1})]
continuous_cols=[c for c in _df.columns if c not in binary_cols and _df[c].nunique()>10]

print('binary cols', binary_cols)
print('continuous cols', continuous_cols)

for b in binary_cols:
    if b=='bad_history':
        continue
    print('\n==', b, 'mean', _df[b].mean(), '==')
    for cont in continuous_cols:
        # compute mean difference (group1 - group0)
        m1=_df.loc[_df[b]==1, cont].mean()
        m0=_df.loc[_df[b]==0, cont].mean()
        print(f"  {cont}: mean1={m1:.3f}, mean0={m0:.3f}, diff={m1-m0:.3f}")

