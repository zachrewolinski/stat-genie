import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

for col in df.columns:
    s = df[col]
    print('\n', col)
    print('dtype', s.dtype, 'nunique', s.nunique())
    if s.dtype != 'object':
        print('min', s.min(), 'max', s.max())
        print('unique sorted sample', sorted(s.unique())[:15])
        print('value counts top', s.value_counts().head(10))
    else:
        print('value counts', s.value_counts())
