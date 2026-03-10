import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print('shape', df.shape)
# identify binary columns
binary_cols = []
for c in df.columns:
    vals = df[c].dropna().unique()
    if set(vals).issubset({0,1}):
        binary_cols.append(c)

print('binary cols', binary_cols)
print('\nBinary col means:')
for c in binary_cols:
    print(c, df[c].mean())

# correlation with each other for binary
print('\nPairwise correlation with deny and accept if present:')
for target in ['deny','accept']:
    if target in df.columns:
        print('Target', target)
        for c in binary_cols:
            if c==target: continue
            corr = np.corrcoef(df[c], df[target])[0,1]
            print(c, corr)

# show head
print('\nHead:')
print(df.head())
