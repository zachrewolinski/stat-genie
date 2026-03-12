import pandas as pd
import numpy as np

df = pd.read_csv("amtl.csv")

print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
# count near integers
near_int = np.isclose(df['num_amtl'], np.round(df['num_amtl']), atol=1e-6)
print('fraction near int', near_int.mean())
print('unique num_amtl sample', df['num_amtl'].head(10).tolist())

# check if any negative
print('negative count', (df['num_amtl'] < 0).sum())

# sockets unique
print('sockets unique', sorted(df['sockets'].unique())[:10], '...', len(df['sockets'].unique()))

# possible proportion range if treat num_amtl as count
prop = df['num_amtl'] / df['sockets']
print('prop min/max', prop.min(), prop.max())

# summary by genus
print(df.groupby('genus')['num_amtl'].mean())
print(df.groupby('genus')['num_amtl'].std())

# check if num_amtl is standardized by overall mean/std
print('overall mean/std', df['num_amtl'].mean(), df['num_amtl'].std())

# check correlation with age
print('corr age', df['num_amtl'].corr(df['age']))
