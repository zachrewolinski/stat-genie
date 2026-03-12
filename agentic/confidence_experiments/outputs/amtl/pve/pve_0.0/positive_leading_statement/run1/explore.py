import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets']].describe())
print('num_amtl unique sample:', df['num_amtl'].head(10).tolist())
print('sockets unique:', sorted(df['sockets'].unique())[:20])
print('num_amtl min/max:', df['num_amtl'].min(), df['num_amtl'].max())
print('Any non-integer num_amtl:', (df['num_amtl']%1!=0).any())
print('Any negative num_amtl:', (df['num_amtl']<0).any())
