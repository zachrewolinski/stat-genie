import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

print('rows', len(df))
print(df.head())
print(df[['num_amtl','sockets']].describe())

# check if num_amtl has integer-ish values
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())

# check if num_amtl correlates with sockets
print('corr num_amtl sockets', df['num_amtl'].corr(df['sockets']))

# check distribution by genus
print(df.groupby('genus')['num_amtl'].describe())

# check range of num_amtl by sockets maybe?
print(df.groupby('sockets')['num_amtl'].mean().head())

# check if num_amtl roughly in [-13,13], looks like scaled around 0.

