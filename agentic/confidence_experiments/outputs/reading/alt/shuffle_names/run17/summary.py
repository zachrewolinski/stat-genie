import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print('shape', df.shape)

for col in df.columns:
    s = df[col]
    print('\n', col)
    print('dtype', s.dtype)
    if s.dtype == 'object':
        print('nunique', s.nunique(dropna=True))
        print('sample', s.dropna().unique()[:10])
        # show top counts
        print('value counts', s.value_counts(dropna=True).head())
    else:
        print('nunique', s.nunique(dropna=True))
        print('min', s.min(), 'max', s.max())
        print('mean', s.mean(), 'std', s.std())
        print('quantiles', s.quantile([0.0,0.25,0.5,0.75,1.0]).to_dict())
