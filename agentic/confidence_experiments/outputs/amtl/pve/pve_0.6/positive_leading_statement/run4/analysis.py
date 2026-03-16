import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets']].describe())
print('num_amtl integer?', (df['num_amtl'].dropna() % 1 == 0).all())
print('min num_amtl', df['num_amtl'].min(), 'max', df['num_amtl'].max())
print('sockets unique', sorted(df['sockets'].unique())[:10])
print('genus counts', df['genus'].value_counts())
