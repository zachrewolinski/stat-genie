import pandas as pd
import numpy as np

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print('\nDtypes:')
print(df.dtypes)

# basic stats
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print('\n', col)
        print(df[col].describe())
    else:
        print('\n', col)
        print(df[col].describe(include='all'))
        print('unique sample:', df[col].dropna().unique()[:10])

# check missing counts
print('\nMissing counts:')
print(df.isna().sum())
