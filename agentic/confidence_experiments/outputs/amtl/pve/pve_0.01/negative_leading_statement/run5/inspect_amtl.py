import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df['num_amtl'].describe())
print('unique num_amtl ints?', np.all(np.isclose(df['num_amtl'], np.round(df['num_amtl']))))
print('num_amtl min,max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets unique', sorted(df['sockets'].unique())[:10])
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())

print('num_amtl mean', df['num_amtl'].mean())

print('min rows')
print(df.loc[df['num_amtl'].idxmin()])
print('max rows')
print(df.loc[df['num_amtl'].idxmax()])

