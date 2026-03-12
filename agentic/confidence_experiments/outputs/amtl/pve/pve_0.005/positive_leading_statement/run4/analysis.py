import pandas as pd
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('num rows', len(df))
print('num_amtl unique', df['num_amtl'].nunique())
print('sockets unique', df['sockets'].nunique())
print('num_amtl min max', df['num_amtl'].min(), df['num_amtl'].max())

# Check if num_amtl seems integer
print('num_amtl integer-like proportion', (df['num_amtl'] % 1 == 0).mean())

# Check genus counts
print(df['genus'].value_counts())
