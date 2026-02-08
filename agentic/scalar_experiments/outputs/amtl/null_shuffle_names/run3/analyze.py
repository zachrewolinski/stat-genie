import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())

num_cols = [c for c in df.columns if df[c].dtype != 'object']
print('num cols', num_cols)
for c in num_cols:
    print(c, df[c].min(), df[c].max(), df[c].nunique(), df[c].head(10).tolist())

# check candidate counts
cands = ['genus','age','pop','num_amtl','stdev_age']

# check integerish
for c in cands:
    vals = df[c]
    frac = np.mean(np.isclose(vals, np.round(vals)))
    print(c, 'intish', frac)

# check relationships for integer count candidates
for a in ['genus','age']:
    for b in ['genus','age']:
        if a==b: continue
        le = (df[a] <= df[b]).mean()
        print(f'{a}<= {b}:', le)

# For each pair of candidate count columns, count rows where one > other
for a in ['genus','age']:
    for b in ['genus','age']:
        if a==b: continue
        print(a, b, 'max diff', (df[a]-df[b]).max())

