import pandas as pd
import json

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df['num_amtl'].describe())
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('sockets unique', sorted(df['sockets'].unique())[:10])
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
print('prob_male unique', sorted(df['prob_male'].unique()))
print('age range', df['age'].min(), df['age'].max())
