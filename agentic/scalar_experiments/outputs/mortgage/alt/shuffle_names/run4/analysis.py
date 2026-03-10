import pandas as pd
import numpy as np

path='mortgage.csv'
df=pd.read_csv(path)

print('shape', df.shape)

# identify binary columns
binary_cols=[c for c in df.columns if df[c].dropna().isin([0,1]).all()]
print('binary cols', binary_cols)
print('binary means')
print(df[binary_cols].mean().sort_values())

# check complements for binary columns
for c in binary_cols:
    for d in binary_cols:
        if c>=d:
            continue
        # check if c + d == 1 for all non-missing
        s=df[[c,d]].dropna()
        if np.allclose(s[c]+s[d],1):
            print('complements', c, d)

# show unique counts for near-binary columns
print('nunique')
print(df.nunique())

