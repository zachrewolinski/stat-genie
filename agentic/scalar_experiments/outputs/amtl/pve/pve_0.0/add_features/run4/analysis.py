import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets']].describe())
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('num_amtl int-like', np.all(np.isclose(df['num_amtl'], np.round(df['num_amtl']))))
print('num_amtl min max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets unique', sorted(df['sockets'].unique())[:10], '... total', df['sockets'].nunique())
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
