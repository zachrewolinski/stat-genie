import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print('\ninfo:')
print(df.dtypes)
# sample unique counts
print('\nunique counts:')
print(df.nunique())
# show value counts for small uniques
for col in df.columns:
    nun = df[col].nunique(dropna=False)
    if nun <= 10:
        print('\n', col, df[col].value_counts(dropna=False))
