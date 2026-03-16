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
        summary.append((col, 'num', float(s.min()), float(s.max()), float(s.mean()), float(s.std()), float(s.isna().mean())))
    else:
        summary.append((col, 'cat', int(s.nunique(dropna=True)), float(s.isna().mean())))
print('\nSummary:')
for row in summary:
    print(row)

print('\nSample values:')
for col in df.columns:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()[:5]
        print(col, vals)
