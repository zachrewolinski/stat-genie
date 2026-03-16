import pandas as pd
import numpy as np


df = pd.read_csv('mortgage.csv')
approval_col = 'deny'
controls = [c for c in df.columns if c != approval_col]
X = df[controls].apply(pd.to_numeric, errors='coerce')

# drop rows with missing
X = X.dropna()

# find constant columns
const_cols = [c for c in X.columns if X[c].nunique() <= 1]
print('constant cols', const_cols)

# find duplicate columns
cols = X.columns.tolist()
seen = {}
duplicates = []
for i,c in enumerate(cols):
    key = tuple(X[c].values)
    if key in seen:
        duplicates.append((seen[key], c))
    else:
        seen[key] = c
print('duplicates', duplicates[:10])

# check correlation near 1 or -1
corr = X.corr()
perfect = []
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if np.isfinite(corr.iloc[i,j]) and abs(corr.iloc[i,j])>0.999999:
            perfect.append((cols[i], cols[j], corr.iloc[i,j]))
print('perfect corr', perfect)

# compute rank
rank = np.linalg.matrix_rank(X.values)
print('rank', rank, 'cols', X.shape[1])

