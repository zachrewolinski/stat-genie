import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets']].describe())
print('num_amtl unique sample', df['num_amtl'].dropna().unique()[:10])
print('sockets unique', sorted(df['sockets'].dropna().unique())[:10])
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('num_amtl integer fraction', (df['num_amtl'].dropna() % 1 == 0).mean())
print('sockets integer fraction', (df['sockets'].dropna() % 1 == 0).mean())
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
