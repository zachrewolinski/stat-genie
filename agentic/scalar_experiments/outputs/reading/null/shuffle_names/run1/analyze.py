import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())

summary = []
for col in df.columns:
    s = df[col]
    # determine type
    dtype = s.dtype
    n_unique = s.nunique(dropna=True)
    sample = s.dropna().astype(str).head(5).tolist()
    # numeric stats
    if pd.api.types.is_numeric_dtype(s):
        summary.append((col, 'numeric', n_unique, s.min(), s.max(), s.mean(), s.std(), sample))
    else:
        summary.append((col, 'categorical', n_unique, None, None, None, None, sample))

for item in summary:
    col, kind, n_unique, minv, maxv, meanv, stdv, sample = item
    print('\n', col)
    print('  kind:', kind)
    print('  unique:', n_unique)
    print('  sample:', sample)
    if kind == 'numeric':
        print('  min:', minv, 'max:', maxv, 'mean:', meanv, 'std:', stdv)
