import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

# Check constant columns
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print('constant cols', const_cols)

# Check duplicate columns (exact)
cols = df.columns
seen = {}
for c in cols:
    vals = tuple(df[c].fillna(-999999).tolist())
    if vals in seen:
        print('duplicate columns', seen[vals], c)
    else:
        seen[vals] = c

# Check if any column is exact complement of another (binary)
for i, c1 in enumerate(cols):
    if set(df[c1].dropna().unique()).issubset({0,1}):
        for c2 in cols[i+1:]:
            if set(df[c2].dropna().unique()).issubset({0,1}):
                if (df[c1] == 1 - df[c2]).all():
                    print('complement columns', c1, c2)

# Correlation matrix to see high correlations
numeric = df.select_dtypes(include=[np.number])
corr = numeric.corr()
# find pairs with abs(corr)>0.99
pairs = []
for i, c1 in enumerate(corr.columns):
    for j, c2 in enumerate(corr.columns):
        if j<=i:
            continue
        if abs(corr.loc[c1,c2])>0.99:
            pairs.append((c1, c2, corr.loc[c1,c2]))
print('high corr pairs (>0.99):', pairs)
