import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('rows', len(df))
print(df[['num_amtl','sockets','age','prob_male']].describe())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets unique', sorted(df['sockets'].unique())[:10])
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
