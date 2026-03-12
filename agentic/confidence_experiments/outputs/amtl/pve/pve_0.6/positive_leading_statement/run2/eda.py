import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('rows', len(df))
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('num_amtl integer?', np.all(np.isclose(df['num_amtl'], np.round(df['num_amtl']))))
print('sockets min/max', df['sockets'].min(), df['sockets'].max())
print('sockets unique', sorted(df['sockets'].unique())[:10])
print('genus value counts\n', df['genus'].value_counts())
print('tooth_class value counts\n', df['tooth_class'].value_counts())
print('prob_male unique', sorted(df['prob_male'].unique()))
