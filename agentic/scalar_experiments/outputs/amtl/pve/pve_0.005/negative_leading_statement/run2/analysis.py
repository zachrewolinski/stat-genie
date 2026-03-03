import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets','age','prob_male']].describe())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets min/max', df['sockets'].min(), df['sockets'].max())
print('unique genus', df['genus'].unique())
print('unique tooth_class', df['tooth_class'].unique())
print('num_amtl integer?', (df['num_amtl']%1==0).all())
print('num_amtl negative count', (df['num_amtl']<0).sum())
print('sockets integer?', (df['sockets']%1==0).all())
print('sockets unique', np.sort(df['sockets'].unique())[:10])
