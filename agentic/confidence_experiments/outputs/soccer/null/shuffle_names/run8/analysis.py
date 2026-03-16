import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.dtypes)

# basic stats for numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
print('num cols', len(num_cols))
print(df[num_cols].describe().T[['min','max','mean','std']].sort_values('max'))

# unique counts small
for c in df.columns:
    uniq = df[c].nunique(dropna=False)
    if uniq<=10:
        print('small uniq', c, uniq, sorted(df[c].dropna().unique())[:20])

# show value ranges for all columns
for c in num_cols:
    series = df[c]
    print(c, 'min', series.min(), 'max', series.max(), 'nunique', series.nunique())
