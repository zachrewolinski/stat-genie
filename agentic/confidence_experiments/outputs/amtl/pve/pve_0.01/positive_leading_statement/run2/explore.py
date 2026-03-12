import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df['num_amtl'].describe())
print('num_amtl unique integer?', np.allclose(df['num_amtl'], df['num_amtl'].round()))
print('num_amtl min', df['num_amtl'].min(), 'max', df['num_amtl'].max())
print('sockets min', df['sockets'].min(), 'max', df['sockets'].max())
print('num_amtl negative count', (df['num_amtl']<0).sum())
print('num_amtl integer count', (df['num_amtl'].round()==df['num_amtl']).sum())
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
