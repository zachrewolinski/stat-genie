import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('shape', df.shape)

# summary stats
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(col, df[col].describe())

# check near-integer for feature3 and feature4
for col in ['feature3','feature4']:
    s = df[col]
    frac = np.abs(s - np.round(s))
    print(col, 'max frac', frac.max(), 'mean frac', frac.mean())

# counts for feature1, feature8
print(df['feature1'].value_counts())
print(df['feature8'].value_counts())
