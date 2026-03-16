import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:')
print(df.dtypes)
print('\nUnique counts:')
print(df.nunique())

num_cols = df.select_dtypes(include=[np.number]).columns
print('\nNumeric summary:')
print(df[num_cols].describe().T)

cat_cols = df.select_dtypes(exclude=[np.number]).columns
for c in cat_cols:
    print('\n', c, 'levels:', df[c].unique()[:10], '... total', df[c].nunique())
