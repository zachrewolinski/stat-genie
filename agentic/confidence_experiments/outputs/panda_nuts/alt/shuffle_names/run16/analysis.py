import pandas as pd
import numpy as np

df = pd.read_csv('panda_nuts.csv')
print('shape', df.shape)
print(df.head())
print('\nunique counts')
print(df.nunique())
print('\nvalue counts for categorical columns:')
for col in df.columns:
    if df[col].dtype == 'object':
        print('\n', col)
        print(df[col].value_counts(dropna=False))

print('\ndescribe numeric')
print(df.describe())
