import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print(df.head())
print('\ncolumns', df.columns.tolist())
print('\nshape', df.shape)
print('\n dtypes')
print(df.dtypes)

# show unique counts for each column (small)
print('\nunique counts')
print(df.nunique())

# show sample unique values for each column
for col in df.columns:
    uniques = df[col].dropna().unique()
    if len(uniques) <= 10:
        print(f"\n{col} uniques: {uniques}")
    else:
        print(f"\n{col} sample uniques: {uniques[:5]}")
