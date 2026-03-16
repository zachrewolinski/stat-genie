import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print('shape', df.shape)
print('num_amtl sample', df['num_amtl'].head().tolist())
print('num_amtl unique sample', df['num_amtl'].unique()[:10])
print('sockets unique sample', df['sockets'].unique()[:10])
print('genus unique', df['genus'].unique())
print('tooth_class unique', df['tooth_class'].unique())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets min/max', df['sockets'].min(), df['sockets'].max())
