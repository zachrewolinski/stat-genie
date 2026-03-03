import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df[['num_amtl','sockets']].describe())
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('sockets unique', sorted(df['sockets'].unique())[:10])
print('genus counts', df['genus'].value_counts())
