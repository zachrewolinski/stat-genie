import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

binary_cols = []
for c in df.columns:
    vals = df[c].dropna().unique()
    if set(vals).issubset({0,1}):
        binary_cols.append(c)

print('missing counts:')
print(df[binary_cols].isna().sum())

print('stds:')
print(df[binary_cols].std())

# use pandas corr
corr = df[binary_cols].corr()
print('corr matrix:')
print(corr)
