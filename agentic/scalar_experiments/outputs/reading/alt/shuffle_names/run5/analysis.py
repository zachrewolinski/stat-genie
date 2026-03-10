import pandas as pd
import json
from pathlib import Path

path = Path('reading.csv')
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())

# basic info
print('\nDtypes:')
print(df.dtypes)

# show unique counts for each column (capped to 10 for display)
print('\nUnique counts:')
for col in df.columns:
    print(col, df[col].nunique())

# show sample unique values for low-card columns
print('\nSample values for low-card columns:')
for col in df.columns:
    nun = df[col].nunique()
    if nun <= 10:
        print(col, sorted(df[col].dropna().unique())[:10])

# describe numeric
print('\nNumeric describe:')
print(df.describe(include='number').T)

