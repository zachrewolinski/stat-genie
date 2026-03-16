import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for col in df.columns:
    series = df[col]
    print('\n', col)
    print('dtype', series.dtype)
    print('n_unique', series.nunique(dropna=True), 'missing', series.isna().sum())
    if pd.api.types.is_numeric_dtype(series):
        print('min', series.min(), 'max', series.max(), 'mean', series.mean(), 'std', series.std())
        print('samples', series.dropna().unique()[:10])
    else:
        print('samples', series.dropna().unique()[:10])
