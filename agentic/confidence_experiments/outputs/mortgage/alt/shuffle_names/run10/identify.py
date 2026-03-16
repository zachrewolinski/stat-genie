import pandas as pd
import numpy as np


df = pd.read_csv('mortgage.csv')

# identify binary columns (0/1) ignoring NaN
binary_cols = []
for col in df.columns:
    s = df[col].dropna()
    uniq = set(s.unique())
    if uniq.issubset({0,1}) and len(uniq) <= 2:
        binary_cols.append(col)

print('binary_cols', binary_cols)

# continuous columns for correlation
cont_cols = [c for c in df.columns if c not in binary_cols]
print('cont_cols', cont_cols)

# compute correlations with two ratio-like columns (continuous)
cont_focus = cont_cols

for b in binary_cols:
    print('\nBinary:', b, 'mean', df[b].mean())
    for c in cont_focus:
        corr = df[[b,c]].corr().iloc[0,1]
        print(f'  corr with {c}: {corr:.3f}')

