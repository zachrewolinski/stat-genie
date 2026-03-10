import pandas as pd
import numpy as np

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())

# basic summary
print('\nhead')
print(df.head())

# identify binary columns
binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
print('\nbinary_cols', binary_cols)

# pairwise check for complements among binary cols
complements = []
for i,c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        s = df[c1] + df[c2]
        if s.isin([1]).all():
            complements.append((c1,c2))
print('complements', complements)

# correlations with deny/accept if exist
for target in ['deny','accept']:
    if target in df.columns:
        print('\nTarget', target)
        print('mean', df[target].mean())
        corr = df[binary_cols].corr()[target].sort_values(ascending=False)
        print('corr with binary cols')
        print(corr)

# distribution of all columns
print('\ncolumn means')
print(df.mean(numeric_only=True))
