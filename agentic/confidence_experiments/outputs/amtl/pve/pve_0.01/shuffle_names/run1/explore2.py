import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# check relationships
print('num_amtl vs age: min diff', (df['num_amtl'] - df['age']).min(), 'max diff', (df['num_amtl'] - df['age']).max())
print('genus vs age: min diff', (df['genus'] - df['age']).min(), 'max diff', (df['genus'] - df['age']).max())
print('genus vs num_amtl: min diff', (df['genus'] - df['num_amtl']).min(), 'max diff', (df['genus'] - df['num_amtl']).max())
# check if any column likely proportion between 0 and 1
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print(col, df[col].min(), df[col].max())

# count integer-like values
for col in ['genus','age','pop','num_amtl','stdev_age']:
    if pd.api.types.is_numeric_dtype(df[col]):
        frac = np.mean(np.isclose(df[col], np.round(df[col])))
        print(col, 'fraction integer-like', frac)
