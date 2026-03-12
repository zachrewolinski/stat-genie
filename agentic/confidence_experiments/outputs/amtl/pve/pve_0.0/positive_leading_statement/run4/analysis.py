import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('num_amtl integer?', np.allclose(df['num_amtl'], df['num_amtl'].round()))
print('num_amtl unique', df['num_amtl'].nunique())
print('sockets min/max', df['sockets'].min(), df['sockets'].max())
print('sockets integer?', np.allclose(df['sockets'], df['sockets'].round()))
print('num_amtl <= sockets?', (df['num_amtl'] <= df['sockets']).all())
print('num_amtl >=0?', (df['num_amtl'] >= 0).all())
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
print('prob_male unique', sorted(df['prob_male'].unique()))
