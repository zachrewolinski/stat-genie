import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

summary = []
for col in df.columns:
    s = df[col]
    nunique = s.nunique(dropna=True)
    dtype = s.dtype
    if np.issubdtype(s.dtype, np.number):
        minv = s.min()
        maxv = s.max()
    else:
        minv = None
        maxv = None
    summary.append((col, str(dtype), nunique, minv, maxv))

summary_df = pd.DataFrame(summary, columns=['col','dtype','nunique','min','max'])
print(summary_df.sort_values(['nunique','col']).head(20).to_string(index=False))
print(summary_df.sort_values(['nunique','col']).tail(20).to_string(index=False))

print('\nSmall-unique columns:')
print(summary_df[summary_df['nunique']<=10].sort_values('nunique').to_string(index=False))

candidates = []
for col in df.columns:
    s = df[col]
    if np.issubdtype(s.dtype, np.number):
        if s.min()>=0 and s.max()<=1 and s.nunique()<=10:
            candidates.append(col)
print('\n0-1 range, <=10 unique:', candidates)

print('\nNumeric cols with max<=50 and min>=0:')
for col in df.columns:
    s = df[col]
    if np.issubdtype(s.dtype, np.number):
        if s.nunique()<=60 and s.max()<=50 and s.min()>=0:
            print(col, 'nunique', s.nunique(), 'min', s.min(), 'max', s.max())
