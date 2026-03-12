import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print('\ncolumns:', list(df.columns))

# basic summary
summary = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        summary.append((col, 'num', s.min(), s.max(), s.mean(), s.std(), s.isna().mean()))
    else:
        summary.append((col, 'cat', s.nunique(dropna=True), s.isna().mean()))
print('\nSummary:')
for row in summary:
    print(row)

# show sample unique values for non-numeric columns
print('\nSample values:')
for col in df.columns:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()[:5]
        print(col, vals)

