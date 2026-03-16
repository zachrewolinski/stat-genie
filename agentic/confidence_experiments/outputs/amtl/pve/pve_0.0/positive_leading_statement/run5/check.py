import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets min/max', df['sockets'].min(), df['sockets'].max())
print('num_amtl integer proportion', (df['num_amtl']%1==0).mean())
print('num_amtl within 0..sockets proportion', ((df['num_amtl']>=0)&(df['num_amtl']<=df['sockets'])).mean())
print('num_amtl within 0..1 proportion', ((df['num_amtl']>=0)&(df['num_amtl']<=1)).mean())
print('sockets unique', df['sockets'].unique()[:10])
print('genus counts', df['genus'].value_counts())
