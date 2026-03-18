import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')
print(df.head())
print('\nSummary:')
for col in df.columns:
    series = df[col]
    if series.dtype == 'O':
        print(col, 'dtype=object', 'unique', series.unique()[:10], 'nunique', series.nunique())
    else:
        print(col, 'dtype', series.dtype, 'min', series.min(), 'max', series.max(), 'nunique', series.nunique(), 'mean', series.mean(), 'std', series.std())

