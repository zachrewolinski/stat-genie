import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print('\ninfo:')
print(df.dtypes)
print('\nunique counts:')
print(df.nunique())
for col in df.columns:
    nun = df[col].nunique(dropna=False)
    if nun <= 10:
        print('\n', col)
        print(df[col].value_counts(dropna=False))
