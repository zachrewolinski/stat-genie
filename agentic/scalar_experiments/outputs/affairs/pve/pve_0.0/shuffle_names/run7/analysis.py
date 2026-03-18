import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
print('\nSummary:')
for col in df.columns:
    s = df[col]
    if s.dtype == 'object':
        print(col, 'dtype', s.dtype, 'unique', s.unique()[:10], 'n_unique', s.nunique())
    else:
        print(col, 'dtype', s.dtype, 'min', s.min(), 'max', s.max(), 'n_unique', s.nunique())
        # show unique values if few
        if s.nunique() <= 10:
            print('  values:', sorted(s.unique()))

