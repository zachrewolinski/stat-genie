import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', list(df.columns))
print('dtypes')
print(df.dtypes)
print('\nhead')
print(df.head())

num_cols = df.select_dtypes(include=[np.number]).columns
print('\nnumeric cols', list(num_cols))
print(df[num_cols].describe().T[['min','max','mean','std']])

# small unique numeric columns
for c in num_cols:
    uniq = df[c].nunique(dropna=True)
    if uniq <= 10:
        print('small uniq', c, uniq, sorted(df[c].dropna().unique())[:10])

# 0-1 small set (likely skin tone)
for c in num_cols:
    vals = df[c].dropna()
    if len(vals) and vals.between(0,1).all():
        uniq = sorted(vals.unique())
        if len(uniq) <= 10:
            print('0-1 small', c, uniq)

# integer small counts
for c in num_cols:
    vals = df[c].dropna()
    if len(vals) and np.all(np.isclose(vals, np.round(vals))):
        if vals.max() <= 10:
            print('int <=10', c, sorted(vals.unique())[:20])

# candidate red card columns (counts with small max)
print('\npossible count columns (max<=50)')
for c in num_cols:
    vals = df[c].dropna()
    if len(vals) and vals.max() <= 50 and vals.min() >= 0:
        print(c, 'min', vals.min(), 'max', vals.max(), 'mean', vals.mean())

# check categorical columns unique counts
cat_cols = [c for c in df.columns if c not in num_cols]
print('\ncat cols', cat_cols)
for c in cat_cols:
    print(c, 'nunique', df[c].nunique(dropna=True), 'sample', df[c].dropna().astype(str).unique()[:3])
