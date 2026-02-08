import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('age unique', sorted(df['age'].unique())[:20], '... max', df['age'].max())
print('genus unique', sorted(df['genus'].unique())[:20], '... max', df['genus'].max())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
print('stdev_age range', df['stdev_age'].min(), df['stdev_age'].max())
print('\nAge by sockets (median):')
print(df.groupby('sockets')['age'].median())
print('\nGenus by sockets (median):')
print(df.groupby('sockets')['genus'].median())

print('\nNum_amtl by sockets (median):')
print(df.groupby('sockets')['num_amtl'].median())

print('\nPop by sockets (median):')
print(df.groupby('sockets')['pop'].median())

# check if genus <= age?
print('\nProportion rows genus<=age:', np.mean(df['genus']<=df['age']))
print('Max genus-age diff', (df['genus']-df['age']).max())
print('Min genus-age diff', (df['genus']-df['age']).min())

# check if num_amtl <= pop? etc
print('Proportion num_amtl<=pop:', np.mean(df['num_amtl']<=df['pop']))

# check if num_amtl <= age
print('Proportion num_amtl<=age:', np.mean(df['num_amtl']<=df['age']))

# check if num_amtl is integer-like
print('num_amtl integer-like proportion', np.mean(np.isclose(df['num_amtl'], np.round(df['num_amtl']))))

