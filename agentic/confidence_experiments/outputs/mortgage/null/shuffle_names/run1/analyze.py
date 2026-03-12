import pandas as pd
import json
import numpy as np
from pathlib import Path

df = pd.read_csv('mortgage.csv')
print(df.head())
print(df.describe(include='all').T)
print('columns', df.columns)

# identify binary columns
binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1})]
print('binary cols', binary_cols)

# show value counts for binary cols
for c in binary_cols:
    print(c, df[c].value_counts(dropna=False).to_dict())

# maybe gender is a binary with around 20% female? We'll check proportion
for c in binary_cols:
    prop = df[c].mean()
    print(c, 'mean', prop)

# compute acceptance/denial columns: there might be two complement columns
# Find pairs of binary columns that are perfect complements.
complements = []
for i,c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        if ((df[c1] + df[c2])==1).all():
            complements.append((c1,c2))
print('complement pairs', complements)

# compute correlation between binary cols
print('binary corr')
print(df[binary_cols].corr())

# logistic regression? We'll just show acceptance rate by each binary col.
for c in binary_cols:
    # assume outcome is maybe deny or accept; we don't know which.
    pass

