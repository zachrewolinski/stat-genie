import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('unique sockets', df['sockets'].unique())
print('unique tooth_class', df['tooth_class'].unique())
print('stdev_age unique', sorted(df['stdev_age'].unique()))
print('age min/max', df['age'].min(), df['age'].max())
print('pop min/max', df['pop'].min(), df['pop'].max())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('genus min/max', df['genus'].min(), df['genus'].max())
print('correlation genus-num_amtl', df['genus'].corr(df['num_amtl']))
print('correlation genus-age', df['genus'].corr(df['age']))
print('correlation num_amtl-age', df['num_amtl'].corr(df['age']))
print('correlation pop-age', df['pop'].corr(df['age']))
