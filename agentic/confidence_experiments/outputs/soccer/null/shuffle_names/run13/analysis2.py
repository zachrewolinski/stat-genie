import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)

cols = ['redCards','nIAT','defeats','rater2','player','yellowReds','meanExp','yellowCards']

for col in cols:
    s = df[col]
    print('\n', col, 'dtype', s.dtype, 'min', s.min(), 'max', s.max(), 'mean', s.mean())
    # value counts for small values
    vc = s.value_counts().sort_index()
    # print first 10 values
    print('unique count', s.nunique())
    print('head value counts:')
    print(vc.head(10))
    print('tail value counts:')
    print(vc.tail(10))

# check integer columns
print('\nNon-integer values counts:')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        non_int = ((s % 1)!=0).sum()
        if non_int>0:
            print(col, 'non_int', non_int)

# check candidate for red cards by how rare >0
print('\nProportion >0 for numeric columns:')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        prop = (s>0).mean()
        if prop<0.2:
            print(col, 'prop>0', prop, 'max', s.max())

