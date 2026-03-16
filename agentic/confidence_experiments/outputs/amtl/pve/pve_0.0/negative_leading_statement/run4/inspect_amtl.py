import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets']].describe())
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('sockets unique', sorted(df['sockets'].unique())[:10], '...')
print('genus unique', df['genus'].unique())
