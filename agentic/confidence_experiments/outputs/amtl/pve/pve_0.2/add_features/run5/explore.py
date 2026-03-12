import pandas as pd

_df = pd.read_csv('amtl.csv')
print('shape', _df.shape)
print(_df.head())
print(_df.dtypes)
print(_df['genus'].unique())
print('num_amtl summary', _df['num_amtl'].describe())
print('sockets summary', _df['sockets'].describe())
print('num_amtl integer-ish?', (_df['num_amtl']%1==0).mean())
print('num_amtl min/max', _df['num_amtl'].min(), _df['num_amtl'].max())
print('sockets unique', sorted(_df['sockets'].unique())[:10])
print('tooth_class unique', _df['tooth_class'].unique())
print('prob_male unique', sorted(_df['prob_male'].unique()))
