import pandas as pd
import numpy as np

_df=pd.read_csv('mortgage.csv')

binary_cols=[c for c in _df.columns if set(_df[c].dropna().unique()).issubset({0,1})]
continuous_cols=[c for c in _df.columns if c not in binary_cols and _df[c].nunique()>10]
ordinal_cols=[c for c in _df.columns if c not in binary_cols and _df[c].nunique()<=10]

cols_for_effect=continuous_cols+ordinal_cols

print('cols_for_effect', cols_for_effect)

for b in binary_cols:
    if b=='bad_history':
        continue
    s=_df[[b]+cols_for_effect].dropna()
    if s[b].nunique()<2:
        continue
    # compute standardized mean difference for each col
    total=0
    effects=[]
    for c in cols_for_effect:
        m1=s.loc[s[b]==1, c].mean()
        m0=s.loc[s[b]==0, c].mean()
        sd=s[c].std()
        if sd==0:
            continue
        smd=(m1-m0)/sd
        effects.append((c, smd))
        total+=abs(smd)
    effects_sorted=sorted(effects, key=lambda x: abs(x[1]), reverse=True)
    print('\n', b, 'mean', s[b].mean(), 'total_abs_smd', total)
    for c,smd in effects_sorted:
        print(f"  {c}: smd={smd:.3f}")

