import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())

for col in df.columns:
    s = df[col]
    print('\n', col)
    print('dtype', s.dtype, 'n_unique', s.nunique(dropna=True))
    print('sample', s.dropna().unique()[:5])
    if s.nunique(dropna=True) <= 10:
        print('value_counts')
        print(s.value_counts(dropna=False).head(10))
