import pandas as pd
import itertools

_df=pd.read_csv('mortgage.csv')

binary_cols=[c for c in _df.columns if set(_df[c].dropna().unique()).issubset({0,1})]

pairs=[]
for c1, c2 in itertools.combinations(binary_cols, 2):
    # alignment on non-missing
    s=_df[[c1,c2]].dropna()
    if len(s)==0:
        continue
    agree=(s[c1]==1-s[c2]).mean()
    pairs.append((agree,c1,c2))

pairs_sorted=sorted(pairs, reverse=True)
print('Top complement-like pairs:')
for agree,c1,c2 in pairs_sorted[:10]:
    print(f"{c1} vs {c2}: complement agreement {agree:.3f}")
