import pandas as pd
import numpy as np
from itertools import combinations

path = 'soccer.csv'

df = pd.read_csv(path)

# numeric integer columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
int_cols = [c for c in num_cols if pd.api.types.is_integer_dtype(df[c])]
print('int cols:', int_cols)

# check for combos where a column equals sum of three others for most rows
cands = int_cols.copy()

results = []
for target in cands:
    others = [c for c in cands if c != target]
    for a,b,c in combinations(others, 3):
        s = df[a] + df[b] + df[c]
        match = (s == df[target]).mean()
        if match > 0.8:
            results.append((target, (a,b,c), match))

results = sorted(results, key=lambda x: -x[2])
print('high match sum combos (>0.8):', results[:10])

# show correlations with potential games column (max 47)
maxes = {c: df[c].max() for c in int_cols}
print('maxes', maxes)

