import pandas as pd
import numpy as np

df = pd.read_csv('hurricane.csv')
print(df.head())
print('\nSummary:')
for col in df.columns:
    s = df[col]
    print('\n', col)
    print('dtype', s.dtype)
    if pd.api.types.is_numeric_dtype(s):
        print('min', s.min(), 'max', s.max(), 'mean', s.mean(), 'std', s.std())
        print('n_unique', s.nunique())
    else:
        print('n_unique', s.nunique())
        print('sample', s.dropna().unique()[:5])

print('\nMissing counts:')
print(df.isna().sum())
