import pandas as pd

df = pd.read_csv('amtl.csv')
print('any genus>age', (df['genus']>df['age']).sum())
print('max genus-age', (df['genus']-df['age']).max())

# if genus is num_amtl and age is num_obs, check fraction range
frac = df['genus']/df['age']
print('frac min/max', frac.min(), frac.max())
print('frac head', frac.head())

# if age is num_obs, check by sockets mean
print(df.groupby('sockets')[['genus','age']].mean())
