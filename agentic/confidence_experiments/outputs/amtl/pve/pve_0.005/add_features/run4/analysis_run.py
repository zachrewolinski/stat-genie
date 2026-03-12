import pandas as pd

path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('num_amtl sample', df['num_amtl'].head(10).tolist())
print('sockets sample', df['sockets'].head(10).tolist())
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
print('prob_male unique', sorted(df['prob_male'].dropna().unique())[:10])
