import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df.nunique())
print('tooth_class unique', df['tooth_class'].unique())
print('sockets unique', df['sockets'].unique())
