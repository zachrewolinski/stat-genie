import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print('\nhead')
print(df.head())

print('\ndtypes')
print(df.dtypes)

# unique counts for each column
print('\nunique counts')
for c in df.columns:
    print(c, df[c].nunique(dropna=False))

# for object columns, show sample unique values
print('\nobject samples')
for c in df.columns:
    if df[c].dtype == 'object':
        vals = df[c].dropna().unique()[:10]
        print(c, vals)

# numeric summary
print('\nnumeric summary')
print(df.describe(include=[np.number]).T)
